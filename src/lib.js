class Drawable {
    constructor(label, x, y, width, height, canvas, strokeRGBA, fillRGBA) {
        this.label = label;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.canvasContext = canvas.getContext("2d");
        // Copy RGBA arrays, if given, else set to null
        this.strokeRGBA = (typeof strokeRGBA === "undefined") ? null : strokeRGBA.slice();
        this.fillRGBA = (typeof fillRGBA === "undefined") ? null : fillRGBA.slice();
    }

    draw() {
        const x = this.x;
        const y = this.y;
        const width = this.width;
        const height = this.height;
        const ctx = this.canvasContext;
        assert([x, y, width, height, ctx].every(val => typeof val !== "undefined"), "Drawable instances must always have defined x, y, width, height, and canvasContext.", {name: "Drawable", obj: this});
        if (this.fillRGBA !== null) {
            ctx.fillStyle = "rgba(" + this.fillRGBA.join(',') + ')';
            ctx.fillRect(x, y, width, height);
        }
        if (this.strokeRGBA !== null) {
            ctx.strokeStyle = "rgba(" + this.strokeRGBA.join(',') + ')';
            ctx.strokeRect(x, y, width, height);
        }
    }
}

// Simulated device memory accesses through L2.
// LRU for convenience and neat visual appearance, although Mei and Chu [1] suggest the L2 replacement policy is not LRU
// The cache can be disabled by giving a zero linesCount argument to the constructor.
// In this case, all memory accesses will still be simulated, but no indexes are cached at any point.
class L2Cache {
    constructor(linesCount) {
        // Cached device memory indexes.
        // All are offset by 1, since 0 represents empty cacheline
        this.lines = new Array(linesCount);
        this.ages = new Array(linesCount);
        // Fill with zeros
        this.clear();
        // Amount of words (graphical slots) in one cacheline
        this.lineSize = CONFIG.cache.L2CacheLineSize;
        // Device memory access instructions waiting to return
        // Each key is a memory index that every instruction in the mapped array are waiting for.
        // e.g. if several threads happen to request the same index `i`, we append the duplicate memory accesses to an array at key `i` instead of creating independent duplicates
        this.memoryAccessQueue = new Map;
        // Same but for cached indexes
        this.cacheAccessQueue = new Map;
    }

    align(i) {
        return i - i % this.lineSize;
    }

    // Iterate over values of all memory access queues
    *allQueuedInstructions() {
        yield* this.memoryAccessQueue.values();
        yield* this.cacheAccessQueue.values();
    }

    // Memory transactions can be coalesced into 32, 64 or 128 byte transactions [2].
    // For simplicity, it is assumed that all addressable memory slots are 4 bytes.
    // Then, adjacent indexes can be coalesced in chunks of 8, 16 and 32 indexes.
    // Returns the aligned cache lines
    coalesceMemoryTransactions(instructions) {
        // Reduction 1, align all memory access indexes from all threads to 32 byte lines
        let cacheLines32 = new Map;
        for (let instruction of instructions) {
            const index = instruction.data.index;
            const aligned32 = index - index % 8;
            const line32 = cacheLines32.get(aligned32);
            if (typeof line32 !== "undefined") {
                // 32 byte cache line starting at `aligned32` exists, we can also fetch `index` with that line
                line32.indexes.add(index);
                line32.lineSize = 8;
            } else {
                // Add a 32 byte cache line starting at `aligned32` with `index` as only memory address
                cacheLines32.set(aligned32, {indexes: new Set([index]), lineSize: 8});
            }
        }

        // Reduction 2, align results to 64 bytes
        let cacheLines64 = new Map;
        for (let [aligned32, line32] of cacheLines32.entries()) {
            const aligned64 = aligned32 - aligned32 % 16;
            if (cacheLines64.has(aligned64)) {
                // line32 has already been coalesced with another 32 byte line
                continue;
            }
            // Get 32 byte line left to line32
            const line32left = cacheLines32.get(aligned64);
            if (typeof line32left === "undefined" || !disjoint(line32.indexes, line32left.indexes)) {
                // No disjoint 32 byte lines to merge, add single 32 byte line
                cacheLines64.set(aligned32, line32);
            } else {
                // We can merge two 32 byte cache lines to fetch 64 bytes in one transaction
                const lineIndexes = new Set;
                for (let index of line32.indexes) {
                    lineIndexes.add(index);
                }
                for (let index of line32left.indexes) {
                    lineIndexes.add(index);
                }
                cacheLines64.set(aligned64, {indexes: lineIndexes, lineSize: 16});
            }
        }

        // Reduction 3, align results to 128 bytes
        let cacheLines128 = new Map;
        for (let [aligned64, line64] of cacheLines64.entries()) {
            // NOTE line64 is 64 bytes wide only if it was merged with two 32 byte lines during reduction 2,
            // it can still be 32 bytes if no pair of adjacent 32 byte lines was found
            const aligned128 = aligned64 - aligned64 % 32;
            if (cacheLines128.has(aligned128)) {
                // line64 has already been coalesced with another 64 byte line
                continue;
            }
            // Get 64 byte line left to line64
            const line64left = cacheLines64.get(aligned128);
            if (typeof line64left === "undefined" || !disjoint(line64.indexes, line64left.indexes)) {
                // No disjoint 64 byte lines to merge, add single 64 byte line
                cacheLines128.set(aligned64, line64);
            } else {
                // We can merge two 64 byte cache lines to fetch 128 bytes in one transaction
                const lineIndexes = new Set;
                for (let index of line64.indexes) {
                    lineIndexes.add(index);
                }
                for (let index of line64left.indexes) {
                    lineIndexes.add(index);
                }
                cacheLines128.set(aligned128, {indexes: lineIndexes, lineSize: 32});
            }
        }

        // The amount cache lines is the minimum amount of required memory transactions to fetch all the indexes
        // In the best case, all instructions coalesce to a single 128 byte transaction
        // In the worst case, every instruction coalesce to 32 independent 32 byte transactions
        assert(1 <= cacheLines128.size <= 32, "alignment failed, number of indexes aligned: " + cacheLines128.size + ", which is not in range [1, 32]");
        return cacheLines128;
    }

    coalesceAndMergeCacheLines() {
        // Coalesce all new memory accesses of distinct indexes to cache lines with a width of 32, 64 or 128 bytes.
        // Get a list of memory access instructions for every unique memory index.
        const instructions = Array.from(this.memoryAccessQueue.values(), instructions => instructions[0])
        // Choose only uncoalesced instructions so we wont coalesce twice
        const unCoalesced = instructions.filter(instruction => {
            assert(typeof instruction.data !== "undefined", "attempting to coalesce memory transactions, but instruction does not have data attributes attached", {name: "in L2Cache.coalesceAndMergeCacheLines, bad Instruction", obj: instruction});
            return !instruction.data.coalesced;
        });
        if (unCoalesced.length === 0) {
            return;
        }

        // Simulate memory transaction coalescing by merging adjacent memory access instructions into 32, 64, and 128 byte cache lines
        const newCacheLines = this.coalesceMemoryTransactions(unCoalesced);
        // Simulate coalesced transaction latency reduction by multiplying the latency of one memory access instruction with the amount of coalesced cache lines
        // TODO simulate memory bandwidth limit somewhere here by allowing through only a limited total width of cache lines at any time.
        // Currently, we assume that all cache lines are loaded in unison and increasing the amount of cache lines just looks like all lines become slower
        const coalescedLatency = newCacheLines.size * unCoalesced[0].cyclesLeft;
        assert(coalescedLatency > 0 && "coalesced memory access latency must be positive");

        // The memory access queue currently has instructions for fetching single indexes from DRAM.
        // Now, add every index from the coalesced cache lines as new instructions, but set the latency for all to a single, reduced latency
        for (let [index, line] of newCacheLines.entries()) {
            for (let i = index; i < index + line.lineSize; ++i) {
                // Assign the total latency to device memory access instructions for every thread
                // NOTE do not overwrite existing deviceMemoryAccess instructions since they are the only reference Thread instances have for checking if their memory accesses have completed
                if (!this.memoryAccessQueue.has(i)) {
                    this.memoryAccessQueue.set(i, new Array);
                }
                let queued = this.memoryAccessQueue.get(i);
                queued.push(Instruction.deviceMemoryAccess(i));
                for (let instruction of queued) {
                    instruction.setLatency(coalescedLatency);
                    instruction.data.coalesced = true;
                }
            }
        }
    }

    // Add fetched data from every completed memory access as 32 byte cache lines
    updateCacheLines() {
        let completedInstructions = new Array;
        for (const [index, instructions] of this.memoryAccessQueue.entries()) {
            for (let instruction of instructions) {
                if (instruction.isDone()) {
                    // Memory access complete, add memory index as cache line
                    const memoryIndex = instruction.data.index;
                    assert(memoryIndex === index, "unexpected index in memory access instruction, expected to be equal to memoryAccessQueue key " + index + ", but was " + memoryIndex);
                    const lineIndex = this.getCachedIndex(index);
                    if (lineIndex < 0) {
                        this.addNew(index);
                    } else {
                        // Index is already in cache, reset line age if the cache is enabled
                        if (this.ages.length > 0) {
                            this.ages[lineIndex] = 0;
                        }
                    }
                    completedInstructions.push(instruction);
                }
            }
        }
        return completedInstructions;
    }

    step() {
        // Get all device memory instructions that have completed waiting
        let completedInstructions = this.updateCacheLines();

        // Add completed cache accesses (for deletion)
        for (const instructions of this.cacheAccessQueue.values()) {
            for (const instruction of instructions) {
                if (instruction.isDone()) {
                    completedInstructions.push(instruction);
                }
            }
        }

        // Drop all completed memory access instructions
        for (let instruction of completedInstructions) {
            assert(instruction.name.startsWith("device") || instruction.name.startsWith("cache"), "unknown memory instruction scheduled for deletion, should be device or cache access", {name: "Instruction", obj: instruction});
            assert(instruction.isDone(), "about to drop incomplete instruction, should only drop completed instructions", {name: "L2Cache.step, Instruction", obj: instruction});
            if (instruction.name.startsWith("device")) {
                this.memoryAccessQueue.delete(instruction.data.index);
            } else {
                this.cacheAccessQueue.delete(instruction.data.index);
            }
        }

        this.coalesceAndMergeCacheLines()

        // Wait one cycle for all memory access instructions
        for (const instructions of this.allQueuedInstructions()) {
            assert(instructions.every(instr => !instr.isDone()), "memory access queues cannot contain finished memory access instructions before advancing the cycle counter");
            for (const instruction of instructions) {
                instruction.cycle();
            }
        }

        // Age all cache lines
        for (let i = 0; i < this.ages.length; ++i) {
            ++this.ages[i];
        }

        // Filter away all memory access indexes that simulate coalescing memory transactions
        // i.e. return all completed cached and device memory instructions that were requested by some thread
        return completedInstructions.filter(instruction => {
            const SM_ID = instruction.data.SM_ID;
            if (typeof SM_ID !== "undefined") assert(SM_ID > 0, "SM_IDs should all start at 1", {name: "during filtering of L2 completed memory instructions, Instruction", obj: instruction});
            return typeof SM_ID !== "undefined";
        });
    }

    getCachedIndex(i) {
        const aligned = this.align(i) + 1;
        return this.lines.findIndex(cached => cached > 0 && aligned === cached);
    }

    addLine(i, j) {
        // Add new cacheline i with cached index j
        this.lines[i] = this.align(j) + 1;
        this.ages[i] = 0;
    }

    // Replace the oldest cacheline with i
    // In case two cache lines have the same age, the line with a lower index is taken as the older one
    // This is only to make the animation a bit prettier
    addNew(i) {
        if (this.lines.length === 0) {
            // Cache is disabled
            return;
        }
        const oldestIndex = this.lines.reduce((oldest, _, index) => {
            if ((this.ages[index] > this.ages[oldest])
                || (this.ages[index] === this.ages[oldest]
                    && index < oldest)) {
                // Cacheline at 'index' is older than line at 'oldest'
                return index;
            } else {
                return oldest;
            }
        }, 0);
        this.addLine(oldestIndex, i);
    }

    // When we already know DRAM index i is not cached, we can queue a device memory access
    queueMemoryAccess(i, SM_ID) {
        // Every memory access will be recorded, even if a cache line retrieval is already pending to that index
        let newInstruction = Instruction.deviceMemoryAccess(i, SM_ID);
        if (!this.memoryAccessQueue.has(i)) {
            // We are not currently waiting for a memory access to index i
            this.memoryAccessQueue.set(i, new Array);
        } else {
            // Memory access at index i already pending, we can reduce the latency of the new instruction to equal that of the oldest pending instruction
            // This means they will complete at the same time
            const oldestInstruction = this.memoryAccessQueue.get(i)[0];
            newInstruction.setLatency(oldestInstruction.cyclesLeft);
        }
        this.memoryAccessQueue.get(i).push(newInstruction);
        return newInstruction;
    }

    queueCacheAccess(i, SM_ID) {
        let newInstruction = Instruction.cachedMemoryAccess(i, SM_ID);
        if (!this.cacheAccessQueue.has(i)) {
            this.cacheAccessQueue.set(i, new Array);
        } else {
            const oldestInstruction = this.cacheAccessQueue.get(i)[0];
            newInstruction.setLatency(oldestInstruction.cyclesLeft);
        }
        this.cacheAccessQueue.get(i).push(newInstruction);
        return newInstruction;
    }

    // Simulate a memory access through L2, return true if the index was in the cache.
    // Also update the LRU age of the line i belongs to.
    fetch(SM_ID, i) {
        let fetchInstruction;
        const j = this.getCachedIndex(i);
        if (j < 0) {
            // i was not cached, create memory fetch and add to queue
            fetchInstruction = this.queueMemoryAccess(i, SM_ID);
        } else {
            // i was cached, set age of i's cacheline to zero
            this.ages[j] = 0;
            fetchInstruction = this.queueCacheAccess(i, SM_ID);
        }
        return fetchInstruction;
    }

    getCacheState(i) {
        if (this.getCachedIndex(i) >= 0) {
            // DRAM memory index i is currently cached
            return {type: "cached", completeRatio: 1.0};
        }
        // DRAM memory index i is not currently cached, check if we have a memory access instruction waiting for it
        let queuedInstructions = this.memoryAccessQueue.get(i);
        if (typeof queuedInstructions !== "undefined") {
            // Multiple memory accesses to a single index are coalesced to a single memory access,
            // thus we can ignore all duplicate instructions and use the first one
            let queuedInstruction = queuedInstructions[0];
            assert(typeof queuedInstruction.completeRatio !== "undefined", "undefined completeratio for instruction", {name: "queued instruction", obj: queuedInstruction});
            // Memory access instruction is pending, return the completion ratio
            return {
                type: "pendingMemoryAccess",
                completeRatio: queuedInstruction.completeRatio
            };
        }
        // DRAM memory index is not in cache and we are not waiting for a memory access
        return {type: "notInCache", completeRatio: 0.0};
    }

    clear() {
        this.lines.fill(0);
        this.ages.fill(0);
    }
}

// Namespace object for kernel simulation and encapsulation of various hardware state.
// CUDA statements should be evaluated with Function.prototype.apply, using a CUDAKernelContext object as an argument.
// As a side effect, this creates an Instruction object into the context object.
class CUDAKernelContext {
    constructor(SM_ID) {
        // Local variables in CUDA kernel namespace
        this.locals = {};
        // Function arguments to kernel
        this.args = {};
        // Thread block indexing as in CUDA
        this.threadIdx = {
            x: null,
            y: null,
        };
        this.blockIdx = {
            x: null,
            y: null,
        };
        this.blockDim = {
            x: null,
            y: null,
        };
        // Store the most recently executed instruction which contains the simulated latency
        this.prevInstruction = null;
        // ID of the SM that has scheduled a warp that executed this context
        this.SM_ID = SM_ID;
        assert(typeof this.SM_ID !== "undefined" && this.SM_ID > 0, "sm ids must be integers, defined in kernel contexts");
    }
    assertDefined(x, f) { assert(typeof x !== "undefined" && !Number.isNaN(x), "Failed to evaluate \"" + f + "\" statement due to undefined kernel context variable. Please check that every variable in every statement is defined."); }

    // Identity function with no latency
    identity(x) {
        this.assertDefined(x, "identity");
        this.prevInstruction = Instruction.identity();
        return x;
    }

    // Simulated arithmetic instruction no latency
    arithmetic(result) {
        this.assertDefined(result, "arithmetic");
        // Create latency
        this.prevInstruction = Instruction.arithmetic();
        return result;
    }

    // Simulated random access memory transaction
    arrayGet(memoryGetHandle, index) {
        this.assertDefined(memoryGetHandle, "arrayGet");
        this.assertDefined(index, "arrayGet");
        this.assertDefined(this.SM_ID, "sm id");
        // Simulate memory get
        this.prevInstruction = memoryGetHandle(index, false, undefined, this.SM_ID);
        // Get the actual value without simulation
        return memoryGetHandle(index, true);
    }

    // Simulated random access memory transaction
    arraySet(memorySetHandle, index, value) {
        this.assertDefined(memorySetHandle, "arraySet");
        this.assertDefined(index, "arraySet");
        this.assertDefined(value, "arraySet");
        // Simulate memory set
        this.prevInstruction = memorySetHandle(index, false, value);
        // Set the actual value without simulation
        memorySetHandle(index, true, value);
    }

    // Jump from current line by offset
    jump(offset) {
        this.assertDefined(offset, "jump");
        this.prevInstruction = Instruction.jump(offset);
    }
}

// Drawable, simulated area of GPU DRAM.
// TODO (multiple drawable instances not yet working)
// Several DeviceMemory instances can be defined, e.g. for representing an input and output array
class DeviceMemory extends Drawable {
    constructor(x, y, canvas, inputDim, slotSize, extraPadding) {
        const rows = inputDim.rows;
        const columns = inputDim.columns;
        const slotPadding = CONFIG.memory.slotPadding;
        const width = columns * slotSize + slotPadding * columns;
        let height = rows * slotSize + slotPadding * rows;
        if (typeof extraPadding !== "undefined") {
            height += extraPadding.amount - slotPadding;
        }
        super("device-memory", x, y, width, height, canvas);
        const slotFillRGBA = CONFIG.memory.slotFillRGBA.slice();
        this.slots = Array.from(
            new Array(columns * rows),
            (_, i) => {
                const slotX = x + (i % columns) * (slotSize + slotPadding);
                const rowIndex = Math.floor(i / columns);
                let slotY = y + rowIndex * (slotSize + slotPadding);
                // Offset all rows that are below the row with extra padding
                if (typeof extraPadding !== "undefined" && extraPadding.index < rowIndex) {
                    slotY += extraPadding.amount;
                }
                // Drawable memory slot
                const memorySlot = new MemorySlot(i, 2, "memory-slot", slotX, slotY, slotSize, slotSize, canvas, undefined, slotFillRGBA);
                // Drawable overlays of different colors on top of the slot, one for each SM
                const overlays = Array.from(
                    CONFIG.animation.SMColorPalette,
                    SM_color => {
                        const coolDownPeriod = CONFIG.memory.coolDownPeriod;
                        const coolDownStep = (1.0 - SM_color[3]) / (coolDownPeriod + 1);
                        return {
                            drawable: new Drawable("memory-slot-overlay-SM-color", slotX, slotY, slotSize, slotSize, canvas, undefined, SM_color),
                            defaultColor: SM_color.slice(),
                            hotness: 0,
                            coolDownPeriod: coolDownPeriod,
                            coolDownStep: coolDownStep,
                        };
                    }
                );
                // Counter indexed by SM ids, counting how many threads of that SM is currently waiting for a memory access to complete from this memory slot i.e. DRAM index
                let threadAccessCounter = new Array(CONFIG.SM.count.max);
                threadAccessCounter.fill(0);
                return {
                    memory: memorySlot,
                    overlays: overlays,
                    threadAccessCounter: threadAccessCounter,
                };
            }
        );
    }

    // Simulated memory access to DRAM index `memoryIndex` by an SM with id `SM_ID`
    touch(SM_ID, memoryIndex) {
        assert(typeof memoryIndex !== "undefined", "memoryIndex must be defined when touching memory slot");
        assert(typeof SM_ID !== "undefined", "SM_ID must be defined when touching memory slot");
        assert(CONFIG.SM.count.min <= SM_ID <= CONFIG.SM.count.max, "attempting to touch a DRAM index " + memoryIndex + " with multiprocessor ID " + SM_ID + " which is out of range of minimum and maximum amount of SMs");
        const slot = this.slots[memoryIndex];
        ++slot.threadAccessCounter[SM_ID - 1];
        let overlay = slot.overlays[SM_ID - 1];
        overlay.hotness = overlay.coolDownPeriod;
        overlay.drawable.fillRGBA[3] = 0.8;
    }

    // When there is no longer a DRAM access instruction to `memoryIndex` by SM with id `SM_ID`, clear overlay from memory slot
    untouch(SM_ID, memoryIndex) {
        assert(typeof memoryIndex !== "undefined", "memoryIndex must be defined when untouching memory slot");
        assert(typeof SM_ID !== "undefined", "SM_ID must be defined when untouching memory slot");
        const queue = this.slots[memoryIndex].threadAccessCounter;
        assert(typeof queue !== "undefined", "every memory slot must have a threadAccessCounter defined");
        if (queue[SM_ID - 1] > 0) {
            --queue[SM_ID - 1];
        }
    }

    step(getCacheState) {
        for (let [i, slot] of this.slots.entries()) {
            // Update state of this memory slot using the L2Cache state handle
            slot.memory.setCachedState(getCacheState(i));
            slot.memory.draw();
        }
        this.draw();
    }

    // Assuming SM_ID integers in range(1, CONFIG.SM.count.max + 1),
    // Generator that yields [index, SM_ID], where index is the enumeration of the generator
    *SMsCurrentlyAccessing(slot) {
        let index = 0;
        const counter = slot.threadAccessCounter;
        for (let SM_ID = 1; SM_ID < counter.length + 1; ++SM_ID) {
            if (counter[SM_ID - 1] > 0) {
                yield [index++, SM_ID];
            }
        }
    }

    numSMsCurrentlyAccessing(slot) {
        return Array.from(this.SMsCurrentlyAccessing(slot)).length;
    }

    draw() {
        for (let [i, slot] of this.slots.entries()) {
            // On top of the memory slot, draw unique color for each SM currently accessing this memory slot
            // Also stack colors horizontally to avoid overlap
            for (let [SM_index, SM_ID] of this.SMsCurrentlyAccessing(slot)) {
                const overlay = slot.overlays[SM_ID - 1];
                const drawable = overlay.drawable;
                assert(typeof drawable !== "undefined", "If an SM touched a memory index, the SM must have some overlay color defined");
                // Save original size
                const originalX = drawable.x;
                const originalWidth = drawable.width;
                // Draw small slice of original so that all slices fit in the slot
                const numSMs = this.numSMsCurrentlyAccessing(slot);
                drawable.x += (numSMs - SM_index - 1) * originalWidth / numSMs;
                drawable.width /= numSMs;
                drawable.draw();
                // Put back original size
                drawable.x = originalX;
                drawable.width = originalWidth;
                // Reduce overlay hotness
                if (overlay.hotness > 0) {
                    --overlay.hotness;
                    // Do not reduce below alpha of default color
                    drawable.fillRGBA[3] = Math.max(overlay.defaultColor[3], drawable.fillRGBA[3] - overlay.coolDownStep);
                }
            }
        }
        super.draw();
    }

    clear() {
        for (let slot of this.slots) {
            slot.memory.clear();
            slot.threadAccessCounter.fill(0);
            slot.overlays.forEach(o => o.hotness = 0);
        }
    }
}

// One memory slot represents a single address in RAM that holds a single 4-byte word
class MemorySlot extends Drawable {
    constructor(index, value, ...drawableArgs) {
        super(...drawableArgs);
        this.index = index;
        this.value = value;
        this.defaultColor = this.fillRGBA.slice();
        this.cachedColor = CONFIG.cache.cachedStateRGBA.slice();
    }

    // Update slot cache status to highlight cached slots in rendering
    setCachedState(state) {
        assert(["cached", "pendingMemoryAccess", "notInCache"].some(t => t === state.type), "Unknown cache state", {name: "cache state", obj: state});
        if (state.type === "pendingMemoryAccess") {
            this.fillRGBA = this.defaultColor;
            this.progressRGBA = this.cachedColor;
            this.progressRatio = state.completeRatio;
        } else {
            this.progressRGBA = null;
            this.progressRatio = null;
        }

        if (state.type === "cached") {
            this.fillRGBA = this.cachedColor;
        } else if (state.type === "notInCache") {
            this.fillRGBA = this.defaultColor;
        }
    }

    draw() {
        // Draw memory slot rectangle
        super.draw();
        // Then, draw memory access progress on top of it
        const x = this.x;
        const y = this.y;
        const width = this.width;
        const height = this.height;
        const ctx = this.canvasContext;
        if (this.progressRGBA !== null) {
            assert(typeof this.progressRatio  !== "undefined" && this.progressRatio !== null && this.progressRatio >= 0 - 1e-6 && this.progressRatio <= 1.0 + 1e-6, "if progressRGBA is given, the progressRatio must also be given and be in range [0, 1]", {name: "drawable.draw progress", obj: {color: this.progressRGBA, ratio: this.progressRatio}});
            ctx.fillStyle = "rgba(" + this.progressRGBA.join(',') + ')';
            const yOffset = (1.0 - this.progressRatio) * height;
            ctx.fillRect(x, y + yOffset, width, height - yOffset);
        }
    }

    clear() {
        this.fillRGBA = this.defaultColor.slice();
    }
}

// Grid of blocks
class Grid {
    constructor(dimGrid, dimBlock) {
        this.dimGrid = dimGrid;
        this.dimBlock = dimBlock;
        this.blocks = Array.from(
            new Array(dimGrid.x * dimGrid.y),
            (_, i) => {
                const x = i % dimGrid.x;
                const y = Math.floor(i / dimGrid.x);
                return new Block({x: x, y: y}, dimBlock);
            });
    }

    nextFreeBlockIndex() {
        return this.blocks.findIndex(block => !block.locked && !block.processed);
    }
}

// Block of threads
class Block {
    constructor(blockIdx, dimBlock) {
        this.locked = false;
        this.processed = false;
        this.dim = dimBlock;
        this.idx = blockIdx;
    }

    // Generator of thread warps from this block, takes as argument a kernel call argument object and the id of an SM that has scheduled this block
    *asWarps(kernelArgs, SM_ID) {
        const warpSize = CONFIG.SM.warpSize;
        const threadCount = this.dim.x * this.dim.y;
        assert(threadCount % warpSize === 0, "Uneven block size, unable to divide block evenly into warps");
        let threadIndexes = [];
        for (let j = 0; j < this.dim.y; ++j) {
            for (let i = 0; i < this.dim.x; ++i) {
                threadIndexes.push({x: i, y: j});
                if (threadIndexes.length === warpSize) {
                    const warp = new Warp(this, threadIndexes.slice(), kernelArgs, SM_ID);
                    threadIndexes = [];
                    yield warp;
                }
            }
        }
    }
}

// Simulated CUDA thread
class Thread {
    constructor(threadIdx) {
        this.idx = threadIdx;
        this.isMasked = false;
        this.statement = null;
        this.kernelContext = null;
        this.instruction = Instruction.empty();
        this.prevInstruction = null;
    }

    // True if this thread is ready to take a new instruction
    isActive() {
        return this.instruction.isDone();
    }

    cycle() {
        assert(typeof this.instruction !== "undefined" && this.instruction !== null, "threads should never have undefined instructions");
        if (this.isMasked) {
            return;
        }
        if (this.statement !== null) {
            assert(this.instruction.isDone(), "thread was assigned a new statement but it still had an instruction with cycles left", {name: "thread", obj: this});
            // This thread has been assigned to execute a new statement, create instruction from queued statement
            // We use a try-catch because the statements can be arbitrary js-functions that can throw just about anything
            try {
                this.statement.apply(this.kernelContext);
            } catch(error) {
                console.error("ERROR: while applying kernel context", this.kernelContext, "to statement", this.statement);
                console.error(error);
                failHard();
            }
            this.prevInstruction = this.instruction;
            this.instruction = this.kernelContext.prevInstruction;
            this.statement = null;
        } else {
            // Continue waiting for instruction to complete
            // Do not advance cycle counters for DRAM access instructions since they are controlled by the L2Cache
            if (this.instruction.name !== "deviceMemoryAccess") {
                this.instruction.cycle();
            }
        }
    }
}

// Simulated group of CUDA threads running in an SM specified by id SM_ID
class Warp {
    constructor(block, threadIndexes, kernelArgs, SM_ID) {
        this.terminated = false;
        this.running = false;
        this.threads = Array.from(threadIndexes, idx => new Thread(idx));
        this.initCUDAKernelContext(block, kernelArgs, SM_ID);
        // Assuming pre-Volta architecture, with program counters for each warp but not yet for each thread
        this.programCounter = 0;
    }

    // An active warp is ready to execute the next instruction
    isActive() {
        return this.threads.every(t => t.isActive());
    }

    // Set threadIdx, blockIdx etc. namespace for simulated CUDA kernel
    initCUDAKernelContext(block, kernelArgs, SM_ID) {
        assert(typeof SM_ID !== "undefined" && 1 <= SM_ID, "cuda kernel contexts must know the SM id which is executing the context, but it was " + SM_ID);
        // Populate simulated CUDA kernel namespace for each thread
        const warpContext = {
            blockIdx: block.idx,
            blockDim: block.dim,
            args: Object.assign({}, kernelArgs),
        };
        for (let t of this.threads) {
            const threadContext = {
                threadIdx: t.idx,
            };
            // Create empty context, overwrite by warp context, then overwrite by thread context, producing the final kernel context for a thread
            t.kernelContext = Object.assign(new CUDAKernelContext(SM_ID), warpContext, threadContext);
        }
    }

    // Add a new statement/instruction to be executed by all threads in the warp
    nextStatement(statement) {
        this.threads.forEach(t => t.statement = statement);
        ++this.programCounter;
    }

    // All threads in a warp do one cycle in parallel
    cycle() {
        this.threads.forEach(t => t.cycle());
        const instr = this.threads[0].instruction;
        if (instr !== null && !instr.isDone()) {
            // Warp wide hacks
            switch (instr.name) {
                // Inelegant jump instruction hack, assuming all threads have the same jump instruction at the same cycle
                case "jump":
                    assert(this.threads.every(t => t.instruction.name === "jump"), "only warp wide jump instructions supported");
                    this.programCounter += instr.data.jumpOffset + Math.sign(instr.data.jumpOffset);
                    break;
            }
        }
        if (instr.cyclesLeft < 0) {
            // negative cycles is a hack for zero-latency instructions, which can be executed an arbitrary amount times within one SM cycle
            assert(this.threads.every(t => t.instruction.cyclesLeft < 0), "either all or no threads in a warp have a zero-latency instruction");
            // Manually override instruction latency,
            // this is to signal the warp simulation it can execute another instruction within the current cycle
            this.threads.forEach(t => t.instruction.cyclesLeft = 0);
            return false;
        }
        return true;
    }
}

// Simulated latency after an SM issued instruction
// latency > 0 : instruction that requires 'latency' amount of SM cycles to complete
// latency = 0 : instruction is complete
// latency < 0 : instruction is a zero latency instruction, which can be executed an arbitrary amount of times within an SM cycle
// TODO simulate memory bandwidth limit by reducing cycle counters of only a limited amount of memory access instructions once
// TODO when threads are waiting for a memory access instruction to complete, the control should be in L2Cache or DeviceMemory
class Instruction {
    constructor(name, latency, data) {
        this.name = name;
        this.data = data || {};
        this.cyclesLeft = null;
        // completeRatio is 0 if cyclesLeft = latency, and 1 if cyclesLeft = 0
        this.completeRatio = null;
        // 1 / latency
        this.completeRatioStep = null;
        this.setLatency(latency);
    }

    isDone() {
        return this.cyclesLeft === 0;
    }

    cycle() {
        if (this.cyclesLeft > 0) {
            --this.cyclesLeft;
            if (this.cyclesLeft === 0) {
                this.completeRatio = 1.0;
            } else {
                this.completeRatio += this.completeRatioStep;
            }
        }
    }

    setLatency(latency) {
        this.cyclesLeft = latency || 0;
        assert(this.cyclesLeft >= -1, "invalid latency value", {name: "Instruction", obj: this});
        this.maxCycles = this.cyclesLeft;
        if (this.maxCycles > 0) {
            this.completeRatio = 0.0;
            this.completeRatioStep = 1.0 / this.maxCycles;
        } else {
            this.completeRatio = 1.0;
            this.completeRatioStep = 0.0;
        }
    }

    // Dummy instruction with zero latency
    static empty() {
        return new Instruction("empty", 0);
    }

    // Identity instruction with no latency, which is distinct from zero latency in that warps can execute an arbitrary amount of no latency instructions per cycle
    static identity() {
        return new Instruction("identity", -1);
    }

    static jump(offset) {
        return new Instruction("jump", -1, {jumpOffset: offset});
    }

    static arithmetic() {
        return new Instruction("arithmetic", CONFIG.latencies[instructionLatencies].arithmetic);
    }

    static cachedMemoryAccess(i, SM_ID) {
        return new Instruction("cachedMemoryAccess", CONFIG.latencies[instructionLatencies].L2CacheAccess, {index: i, SM_ID: SM_ID});
    }

    static deviceMemoryAccess(i, SM_ID) {
        return new Instruction("deviceMemoryAccess", CONFIG.latencies[instructionLatencies].memoryAccess, {index: i, SM_ID: SM_ID});
    }
}

class SMstats {
    constructor(SM_ID) {
        this.stateElement = document.getElementById("sm-state-" + SM_ID);
        this.cycleCounter = this.stateElement.querySelector("ul li pre span.sm-cycle-counter");
        this.blockIdxSpan = this.stateElement.querySelector("ul li pre span.sm-current-block-idx");
        this.cycles = 0;
        this.setColor(CONFIG.animation.SMColorPalette[SM_ID - 1]);
    }

    cycle() {
        ++this.cycles;
        this.cycleCounter.innerHTML = this.cycles;
    }

    setActiveBlock(block) {
        this.blockIdxSpan.innerHTML = block ? "(x: " + block.idx.x + ", y:" + block.idx.y + ")" : '&ltnone&gt';
    }

    terminate() {
        this.setColor([200, 200, 200, 0.25]);
        this.setActiveBlock(null);
    }

    setColor(color) {
        this.stateElement.style.backgroundColor = "rgba(" + color.join(',') + ')';
    }
}

class SMController {
    constructor(SM_ID) {
        this.schedulerCount = CONFIG.SM.warpSchedulers;
        this.residentWarps = [];
        this.grid = null;
        this.program = null;
        this.activeBlock = null;
        this.kernelArgs = null;
        this.statsWidget = new SMstats(SM_ID);
        this.SM_ID = SM_ID;
    }

    // Free all resident warps and take next available block from the grid
    scheduleNextBlock() {
        const i = this.grid.nextFreeBlockIndex();
        if (i < 0) {
            // No free blocks
            return false;
        }
        const newBlock = this.grid.blocks[i];
        this.activeBlock = newBlock;
        this.activeBlock.locked = true;
        this.residentWarps = Array.from(this.activeBlock.asWarps(this.kernelArgs, this.SM_ID));
        this.statsWidget.setActiveBlock(this.activeBlock);
        return true;
    }

    releaseProcessedBlock() {
        if (this.activeBlock !== null) {
            this.activeBlock.locked = false;
            this.activeBlock.processed = true;
            this.activeBlock = null;
        }
    }

    // Return a generator of all non-terminated warps
    *nonTerminatedWarps() {
        for (let warp of this.residentWarps)
            if (!warp.terminated)
                yield warp;
    }

    // Return a generator of all active warps
    *activeWarps() {
        for (let warp of this.nonTerminatedWarps())
            if (warp.isActive())
                yield warp;
    }

    // Return a generator of all free warps, available for scheduling
    *freeWarps() {
        for (let warp of this.activeWarps())
            if (!warp.running)
                yield warp;
    }

    // Return a generator of all scheduled warps
    *scheduledWarps() {
        for (let warp of this.nonTerminatedWarps())
            if (warp.running)
                yield warp;
    }

    // Return a generator of scheduled warps waiting for their instructions to complete
    *blockingWarps() {
        for (let warp of this.scheduledWarps())
            if (!warp.isActive())
                yield warp;
    }

    nextFreeWarp() {
        return this.freeWarps().next().value;
    }

    nextBlockingWarp() {
        return this.blockingWarps().next().value;
    }

    hasNonTerminatedWarps() {
        return !this.nonTerminatedWarps().next().done;
    }

    generatorLength(generator) {
        let count = 0;
        for (let _ of generator) {
            ++count;
        }
        return count;
    }

    scheduledWarpsCount() {
        return this.generatorLength(this.scheduledWarps());
    }

    activeWarpsCount() {
        return this.generatorLength(this.activeWarps());
    }

    // Replace waiting warps with active warps
    scheduleWarps() {
        let assertLoopCount = 0;
        while (this.scheduledWarpsCount() < this.schedulerCount) {
            if (assertLoopCount++ > 100) { assert(false, "failed to schedule next warp", {name: "resident warps", obj: this.residentWarps}); }
            const freeWarp = this.nextFreeWarp();
            if (freeWarp) {
                freeWarp.running = true;
                continue;
            } else {
                const blockingWarp = this.nextBlockingWarp();
                if (blockingWarp) {
                    blockingWarp.running = false;
                    continue;
                }
            }
            // Warp has no free or blocked warps, cannot schedule
            break;
        }
    }

    updateProgramCounters() {
        for (let warp of this.scheduledWarps()) {
            if (!warp.isActive()) {
                // Warp is waiting for instructions to complete.
                // This situation may happen if all resident warps are waiting
                continue;
            }
            const warpPC = warp.programCounter;
            const statements = this.program.statements;
            if (warpPC < statements.length) {
                warp.nextStatement(statements[warpPC]);
            } else {
                warp.terminated = true;
            }
        }
    }

    cycle() {
        this.statsWidget.cycle();
        this.scheduleWarps();
        let isDone = true;
        if (this.hasNonTerminatedWarps()) {
            // Execute warps
            this.updateProgramCounters();
            for (let warp of this.nonTerminatedWarps()) {
                if (!warp.cycle() && isDone) {
                    isDone = false;
                }
            }
        } else {
            // All warps have terminated, try to take next block from grid
            this.releaseProcessedBlock();
            const hasWork = this.scheduleNextBlock();
            if (!hasWork) {
                this.program = null;
            }
        }
        // Return false if this cycle executed a zero latency instruction, meaning the controller should immediately execute the next one
        return isDone;
    }
}

class StreamingMultiprocessor {
    constructor(SM_ID) {
        this.controller = new SMController(SM_ID);
    }

    // Simulate one processor cycle
    cycle() {
        if (this.controller.program !== null) {
            this.controller.cycle();
        }
    }

    // Animation loop step
    step() {
        if (Math.random() < CONFIG.SM.latencyNoiseProb) {
            // Simulated latency within this SM for the duration of a single animation frame
            return;
        }
        this.cycle();
    }

    get_ID() {
        return this.controller.SM_ID;
    }
}

// Wrapper around the device memory and multiprocessors, simulating memory access handling and scheduling
class Device {
    constructor(memoryCanvas, smCount, cacheLines, input, memorySlotSize, extraPadding) {
        this.memory = new DeviceMemory(0, 0, memoryCanvas, input, memorySlotSize, extraPadding);
        this.multiprocessors = this.createProcessors(smCount);
        this.kernelSource = null;
        this.L2Cache = new L2Cache(cacheLines);
        this.kernelHighlightingOn = true;
    }

    // Initialize all processors with new program
    setProgram(grid, program) {
        this.kernelSource = new KernelSource(program.sourceLines, program.sourceLineHeight);
        this.multiprocessors.forEach(sm => {
            assert(sm.controller.program === null, "sm controllers should not be reset while they are running a program");
            sm.controller.kernelArgs = Object.assign({}, program.kernelArgs);
            sm.controller.program = program;
            sm.controller.grid = grid;
            sm.controller.scheduleNextBlock();
        });
    }

    setKernelHighlighting(on) {
        this.kernelHighlightingOn = on;
    }

    programTerminated() {
        return this.multiprocessors.every(sm => sm.controller.program === null);
    }

    createProcessors(count) {
        return Array.from(new Array(count), (_, i) => new StreamingMultiprocessor(i + 1));
    }

    memoryTransaction(type, i, noSimulation, newValue, SM_ID) {
        assert(type === "get" || type === "set", "Invalid memory access type: " + type);
        if (typeof noSimulation !== "undefined" && noSimulation) {
            // Access actual JavaScript array value
            if (type === "get") {
                return this.memory.slots[i].memory.value;
            } else if (type === "set") {
                this.memory.slots[i].memory.value = newValue;
            }
        } else {
            // Simulate memory access through L2Cache and return an Instruction with latency
            if (type === "get" || type === "set") {
                // Touch memory slot i with an SM to trigger visualization
                this.memory.touch(SM_ID, i);
                // Return instruction with latency
                return this.L2Cache.fetch(SM_ID, i);
            }
        }
    }

    step() {
        for (let sm of this.multiprocessors) {
            sm.step();
            // Update line highlights for each warp that has started executing
            for (let warp of sm.controller.nonTerminatedWarps()) {
                const lineno = warp.programCounter;
                if (this.kernelHighlightingOn && lineno > 0) {
                    this.kernelSource.setHighlight(sm.get_ID() - 1, lineno);
                }
            }
        }
        this.kernelSource.step();
        const completedMemoryInstructions = this.L2Cache.step();
        for (let instruction of completedMemoryInstructions) {
            // Clear SM overlay from each memory slot
            this.memory.untouch(instruction.data.SM_ID, instruction.data.index);
        }
        const L2CacheStateHandle = this.L2Cache.getCacheState.bind(this.L2Cache);
        this.memory.step(L2CacheStateHandle);
    }

    // Revert memory cell state colors
    clear() {
        this.memory.clear();
        this.multiprocessors.forEach(sm => sm.controller.statsWidget.terminate());
    }
}

// Kernel source line highlighting with a queue for stacking highlight colors
// according to arrival time
class KernelSource {
    constructor(sourceLines, lineHeight) {
        const palette = CONFIG.animation.SMColorPalette;
        this.highlightedLines = Array.from(sourceLines, (line, lineno) => {
            const x = 0;
            const y = lineno * lineHeight;
            const width = kernelCanvas.width;
            return {
                queue: [],
                colors: Array.from(palette, color => {
                    // Set line highlight alpha lower than SM color
                    let hlColor = color.slice();
                    hlColor[3] *= 0.5;
                    return {
                        drawable: new Drawable("kernel-source-line-highlighting", x, y, width, lineHeight, kernelCanvas, undefined, hlColor),
                    };
                }),
            };
        });
    }

    // Set line with number lineno highlighted with the line's color defined at colorIndex
    setHighlight(colorIndex, lineno) {
        this.highlightedLines[lineno].queue.push(colorIndex);
    }

    drawHighlighted() {
        this.highlightedLines.forEach(line => {
            const stackedColorCount = line.queue.length;
            if (stackedColorCount === 0) {
                // No highlighting on this line, skip
                return;
            }
            // Highlight all queued colors
            line.queue.forEach((colorIndex, queueIndex) => {
                // Offset all colors and render first colors in queue at the bottom
                const drawable = line.colors[colorIndex].drawable;
                const originalY = drawable.y;
                const originalHeight = drawable.height;
                drawable.y += (stackedColorCount - queueIndex - 1) * originalHeight/stackedColorCount
                drawable.height /= stackedColorCount;
                drawable.draw();
                drawable.y = originalY;
                drawable.height = originalHeight;
            });
            line.queue = [];
        });
    }

    step() {
        this.drawHighlighted();
    }
}
