class Drawable {
    constructor(x, y, width, height, canvas, strokeRGBA, fillRGBA, overlayRGBA, overlayRatio) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.canvasContext = canvas.getContext("2d");
        // Copy RGBA arrays, if given, else set to null
        this.strokeRGBA = (typeof strokeRGBA === "undefined") ? null : strokeRGBA.slice();
        this.fillRGBA = (typeof fillRGBA === "undefined") ? null : fillRGBA.slice();
        // If overlayRGBA is given, the drawable will be partially overwritten by a rect of height overlayRatio * this.height, of color overlayRGBA.
        if (typeof overlayRGBA !== "undefined") {
            this.overlayRGBA = overlayRGBA;
            this.overlayRatio = overlayRatio;

        } else {
            this.overlayRGBA = null;
            this.overlayRatio = null;
        }
    }

    draw() {
        const x = this.x;
        const y = this.y;
        const width = this.width;
        const height = this.height;
        const ctx = this.canvasContext;
        if (this.fillRGBA !== null) {
            ctx.fillStyle = "rgba(" + this.fillRGBA.join(',') + ')';
            ctx.fillRect(x, y, width, height);
        }
        if (this.strokeRGBA !== null) {
            ctx.strokeStyle = "rgba(" + this.strokeRGBA.join(',') + ')';
            ctx.strokeRect(x, y, width, height);
        }
        if (this.overlayRGBA !== null) {
            assert(typeof this.overlayRatio  !== "undefined" && this.overlayRatio !== null && this.overlayRatio >= 0 - 1e-6 && this.overlayRatio <= 1.0 + 1e-6, "if overlayRGBA is given, the overlayRatio must also be given and be in range [0, 1]", {name: "drawable.draw overlay", obj: {color: this.overlayRGBA, ratio: this.overlayRatio}});
            ctx.fillStyle = "rgba(" + this.overlayRGBA.join(',') + ')';
            const yOffset = (1.0 - this.overlayRatio) * height;
            ctx.fillRect(x, y + yOffset, width, height - yOffset);
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
    }

    align(i) {
        return i - i % this.lineSize;
    }

    // Memory transactions can be coalesced into 32, 64 or 128 byte transactions [2].
    // For simplicity, it is assumed that all addressable memory slots are 4 bytes.
    // Then, adjacent indexes can be coalesced in chunks of 8, 16 and 32 indexes.
    // Returns the aligned cache lines
    coalesceMemoryTransactions(instructions) {
        // Reduction 1, align all memory access indexes from all threads to 32 byte lines
        //let alignedIndexes = new Set;
        let cacheLines32 = new Map;
        instructions.forEach(instruction => {
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
        });

        // Reduction 2, align results to 64 bytes
        let cacheLines64 = new Map;
        cacheLines32.forEach((line32, aligned32) => {
            const aligned64 = aligned32 - aligned32 % 16;
            const line64 = cacheLines64.get(aligned64);
            if (typeof line64 !== "undefined") {
                // Cache line aligned to `aligned64` exists, we can merge it with another 32 byte cache line to fetch 64 bytes in one transaction
                line32.indexes.forEach(index32 => line64.indexes.add(index32));
                line64.lineSize = 16;
            } else {
                // 64 byte cache line does not exist, using a 32 byte cache line is still sufficient
                cacheLines64.set(aligned64, {indexes: new Set(line32.indexes), lineSize: 8});
            }
        });

        // Reduction 3, align results to 128 bytes
        let cacheLines128 = new Map;
        cacheLines64.forEach((line64, aligned64) => {
            // NOTE line64 is 64 bytes wide only if it was merged with two 32 byte lines during reduction 2,
            // it can still be 32 bytes if no pair of adjacent 32 byte lines was found
            const aligned128 = aligned64 - aligned64 % 32;
            const line128 = cacheLines128.get(aligned128);
            if (typeof line128 !== "undefined") {
                // Cache line aligned to `aligned128` exists, we can merge it with another 64 byte cache line to fetch 128 bytes in one transaction
                line64.indexes.forEach(index64 => line128.indexes.add(index64));
                line128.lineSize = 32;
            } else {
                // 128 byte cache line does not exist, no need to merge yet
                cacheLines128.set(aligned128, {indexes: new Set(line64.indexes), lineSize: line64.lineSize});
            }
        });

        // The amount cache lines is the minimum amount of required memory transactions to fetch all the indexes
        // In the best case, all instructions coalesce to a single 128 byte transaction
        // In the worst case, every instruction coalesce to 32 independent 32 byte transactions
        assert(1 <= cacheLines128.size <= 32, "alignment failed, number of indexes aligned: " + cacheLines128.size + ", which is not in range [1, 32]");
        return cacheLines128;
    }


    step() {
        // Age all cache lines
        for (let i = 0; i < this.ages.length; ++i) {
            ++this.ages[i];
        }

        // Add fetched data from every completed memory access as cache lines
        this.memoryAccessQueue.forEach((instructions, index) => {
            // It is possible to have several memory access instructions to the same index
            instructions.forEach(instruction => {
                if (instruction.isDone()) {
                    // The DRAM index this memory access instruction is waiting data from
                    const memoryIndex = instruction.data.index;
                    assert(memoryIndex === index, "unexpected index in memory access instruction, expected to be equal to memoryAccessQueue key " + index + ", but was " + memoryIndex);
                    // Memory access complete, add cache line
                    const lineIndex = this.getCachedIndex(index);
                    if (lineIndex < 0) {
                        this.addNew(index);
                    } else if (this.ages.length > 0) {
                        // Reset cache line age only if cache is enabled
                        this.ages[lineIndex] = 0;
                    }
                }
            });
        });

        // Drop all completed memory access instructions
        Array.from(this.memoryAccessQueue.keys()).forEach(key => {
            const instructions = this.memoryAccessQueue.get(key);
            if (instructions.some(instr => instr.isDone())) {
                this.memoryAccessQueue.delete(key)
            }
        });

        // Wait one cycle for all memory access instructions
        for (let instructions of this.memoryAccessQueue.values()) {
            instructions.forEach(instr => instr.cycle());
        }

        // Coalesce all new memory accesses of distinct indexes to cache lines with a width of 32, 64 or 128 bytes.
        // Get a list of memory access instructions for every unique memory index.
        const instructions = Array.from(this.memoryAccessQueue.values(), instructions => instructions[0])
        // Check that we do not coalesce twice
        const isCoalesced = instructions.some(instruction => {
            const data = instruction.data;
            assert(typeof data !== "undefined", "attempting to coalesce memory transactions, but instruction does not have data attributes attached", {name: "coalescing instruction", obj: instruction});
            return (typeof data.coalesced !== "undefined" && data.coalesced);
        });
        if (!isCoalesced && instructions.length > 0) {
            // Found new memory access instructions, requested by thread warps
            const newCacheLines = this.coalesceMemoryTransactions(instructions);
            console.log(newCacheLines);
            // Compute the total latency of all transactions
            // TODO simulate memory bandwidth limit here by controlling latency of each cache line independently
            const coalescedLatency = newCacheLines.size * instructions[0].cyclesLeft;
            // Expand the memory access queue with full cache lines
            for (let [index, line] of newCacheLines.entries()) {
                for (let i = index; i < index + line.lineSize; ++i) {
                    // Assign the total latency to device memory access instructions for every thread, create new instructions if needed
                    this.memoryAccessQueue.set(i, new Array);
                    let queued = this.memoryAccessQueue.get(i);
                    queued.push(Instruction.deviceMemoryAccess(i));
                    for (let instruction of queued) {
                        instruction.setLatency(coalescedLatency);
                        instruction.data.coalesced = true;
                    }
                }
            }
        }
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

    queueMemoryAccess(i) {
        // Check if some memory accesses to index i are already pending
        let newInstruction;
        if (this.memoryAccessQueue.has(i)) {
            // Memory access at index i already pending, copy instruction
            newInstruction = Object.assign(Instruction.empty(), this.memoryAccessQueue.get(i)[0]);
        } else {
            // Queue is empty, create new instruction to access memory at index i
            newInstruction = Instruction.deviceMemoryAccess(i);
            this.memoryAccessQueue.set(i, [newInstruction]);
        }
        return newInstruction;
    }

    // Simulate a memory access through L2, return true if the index was in the cache.
    // Also update the LRU age of the line i belongs to.
    fetch(i) {
        let fetchInstruction;
        const j = this.getCachedIndex(i);
        if (j < 0) {
            // i was not cached, create memory fetch and add to queue
            fetchInstruction = this.queueMemoryAccess(i);
        } else {
            // i was cached, set age of i's cacheline to zero
            this.ages[j] = 0;
            fetchInstruction = Instruction.cachedMemoryAccess();
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
            return {type: "pendingMemoryAccess", completeRatio: queuedInstruction.completeRatio};
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
    constructor() {
        this.locals = {};
        this.args = {};
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
        this.prevInstruction = null;
    }

    // one line for easier automatic exclusion
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
        // Simulate memory get
        this.prevInstruction = memoryGetHandle(index);
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
        super(x, y, width, height, canvas);
        const slotFillRGBA = CONFIG.memory.slotFillRGBA;
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
                return new MemorySlot(i, 2, slotX, slotY, slotSize, slotSize, canvas, undefined, slotFillRGBA);
            }
        );
    }

    touch(i) {
        this.slots[i].touch();
    }

    step(getCacheState) {
        this.slots.forEach((slot, i) => {
            slot.setCachedState(getCacheState(i));
            slot.step();
        });
        super.draw();
    }

    clear() {
        this.slots.forEach(slot => slot.clear());
    }
}

// One memory slot represents a single address in RAM that holds a single 4-byte word
class MemorySlot extends Drawable {
    constructor(index, value, ...drawableArgs) {
        super(...drawableArgs);
        this.index = index;
        this.value = value;
        // Copy default color
        this.defaultColor = this.fillRGBA.slice();
        this.cachedColor = CONFIG.cache.cachedStateRGBA.slice();
    }

    // Simulate a memory access to this index
    touch() {
    }

    // Update slot cache status to highlight cached slots in rendering
    setCachedState(state) {
        switch(state.type) {
            case "cached":
                this.fillRGBA = this.cachedColor.slice();
                this.overlayRGBA = null;
                this.overlayRatio = null;
                break;
            case "pendingMemoryAccess":
                this.fillRGBA = this.defaultColor.slice();
                this.overlayRGBA = this.cachedColor.slice();
                this.overlayRatio = state.completeRatio;
                break;
            case "notInCache":
                this.fillRGBA = this.defaultColor.slice();
                this.overlayRGBA = null;
                this.overlayRatio = null;
                break;
            default:
                console.error("unknown cache state: ", state);
                break;
        }
    }

    // Animation step
    step() {
        this.draw();
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

    // Generator of thread warps from this block, takes as argument a kernel call argument object
    *asWarps(kernelArgs) {
        const warpSize = CONFIG.SM.warpSize;
        const threadCount = this.dim.x * this.dim.y;
        if (threadCount % warpSize !== 0) {
            console.error("Uneven block size, unable to divide block evenly into warps");
            return;
        }
        let threadIndexes = [];
        for (let j = 0; j < this.dim.y; ++j) {
            for (let i = 0; i < this.dim.x; ++i) {
                threadIndexes.push({x: i, y: j});
                if (threadIndexes.length === warpSize) {
                    const warp = new Warp(this, threadIndexes.slice(), kernelArgs);
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
    }

    // True if this thread is ready to take a new instruction
    isActive() {
        return this.instruction.isDone();
    }

    cycle() {
        if (this.isMasked) {
            return;
        }
        if (this.statement !== null) {
            // Create new instruction from queued statement
            try {
                this.statement.apply(this.kernelContext);
            } catch(error) {
                console.error("ERROR: while applying kernel context", this.kernelContext, "to statement", this.statement);
                console.error(error);
                failHard();
            }
            this.instruction = this.kernelContext.prevInstruction;
            this.statement = null;
        } else {
            // Continue waiting for instruction to complete
            // Do not advance the cycle counters for memory access instructions since they are controlled by the L2Cache
            if (!this.instruction.name.endsWith("memoryAccess")) {
                this.instruction.cycle();
            }
        }
    }
}

// Simulated group of CUDA threads running in an SM
class Warp {
    constructor(block, threadIndexes, kernelArgs) {
        this.terminated = false;
        this.running = false;
        this.threads = Array.from(threadIndexes, idx => new Thread(idx));
        this.initCUDAKernelContext(block, kernelArgs);
        // Assuming pre-Volta architecture, with program counters for each warp but not yet for each thread
        this.programCounter = 0;
    }

    // An active warp is ready to execute the next instruction
    isActive() {
        return this.threads.every(t => t.isActive());
    }

    // Set threadIdx, blockIdx etc. namespace for simulated CUDA kernel
    initCUDAKernelContext(block, kernelArgs) {
        // Populate simulated CUDA kernel namespace for each thread
        const warpContext = {
            blockIdx: block.idx,
            blockDim: block.dim,
            args: Object.assign({}, kernelArgs),
        };
        this.threads.forEach(t => {
            const threadContext = {
                threadIdx: t.idx,
            };
            t.kernelContext = Object.assign(new CUDAKernelContext(), warpContext, threadContext);
        });
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
        this.setLatency(latency);
    }

    isDone() {
        return this.cyclesLeft === 0;
    }

    cycle() {
        if (this.cyclesLeft > 0) {
            --this.cyclesLeft;
            this.completeRatio += this.completeRatioStep;
        }
    }

    setLatency(latency) {
        this.cyclesLeft = latency || 0;
        assert(this.cyclesLeft >= -1, "invalid latency value", {name: "Instruction", obj: this});
        this.maxCycles = this.cyclesLeft;
        // cyclesLeft in range [0, 1]
        this.completeRatio = 0.0;
        this.completeRatioStep = (this.maxCycles > 0) ? 1.0 / this.maxCycles : 0.0;
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

    static cachedMemoryAccess() {
        return new Instruction("cachedMemoryAccess", CONFIG.latencies[instructionLatencies].L2CacheAccess);
    }

    static deviceMemoryAccess(i) {
        return new Instruction("deviceMemoryAccess", CONFIG.latencies[instructionLatencies].memoryAccess, {index: i});
    }
}

class SMstats {
    constructor(processorID) {
        this.stateElement = document.getElementById("sm-state-" + processorID);
        this.cycleCounter = this.stateElement.querySelector("ul li pre span.sm-cycle-counter");
        this.blockIdxSpan = this.stateElement.querySelector("ul li pre span.sm-current-block-idx");
        this.cycles = 0;
        this.setColor(CONFIG.animation.kernelHighlightPalette[processorID - 1]);
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
    constructor(id) {
        this.schedulerCount = CONFIG.SM.warpSchedulers;
        this.residentWarps = [];
        this.grid = null;
        this.program = null;
        this.activeBlock = null;
        this.kernelArgs = null;
        this.statsWidget = new SMstats(id);
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
        this.residentWarps = Array.from(this.activeBlock.asWarps(this.kernelArgs));
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
    constructor(id) {
        this.controller = new SMController(id);
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

    memoryTransaction(type, i, noSimulation, newValue) {
        if (type !== "get" && type !== "set") {
            console.error("Invalid memory access type:", type);
        }
        if (typeof noSimulation !== "undefined" && noSimulation) {
            // Access actual JavaScript array value
            if (type === "get") {
                return this.memory.slots[i].value;
            } else if (type === "set") {
                this.memory.slots[i].value = newValue;
            }
        } else {
            // Simulate memory access through L2Cache and return an Instruction with latency
            if (type === "get" || type === "set") {
                // Touch memory slot to trigger visualization
                this.memory.touch(i);
                // Return instruction with latency
                return this.L2Cache.fetch(i);
            }
        }
    }

    step() {
        this.multiprocessors.forEach((sm, smIndex) => {
            sm.step();
            // Update line highlights for each warp that has started executing
            for (let warp of sm.controller.nonTerminatedWarps()) {
                const lineno = warp.programCounter;
                if (this.kernelHighlightingOn && lineno > 0) {
                    this.kernelSource.setHighlight(smIndex, lineno);
                }
            }
        });
        this.kernelSource.step();
        this.L2Cache.step();
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
        const palette = CONFIG.animation.kernelHighlightPalette;
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
                        drawable: new Drawable(x, y, width, lineHeight, kernelCanvas, undefined, hlColor),
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
                const prevY = drawable.y;
                const prevHeight = drawable.height;
                drawable.y += (stackedColorCount - queueIndex - 1) * prevHeight/stackedColorCount
                drawable.height /= stackedColorCount;
                drawable.draw();
                drawable.y = prevY;
                drawable.height = prevHeight;
            });
            line.queue = [];
        });
    }

    step() {
        this.drawHighlighted();
    }
}
