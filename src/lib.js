"use strict";

class Drawable {
    constructor(x, y, width, height, canvas, strokeRGBA, fillRGBA) {
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
// LRU for convenience, although Mei and Chu [1] suggest the L2 replacement policy is not LRU
class L2Cache {
    constructor() {
        // Cached device memory indexes.
        // All are offset by 1, since 0 represents empty cacheline
        this.lines = new Uint16Array(CONFIG.cache.L2CacheLines);
        this.ages = new Uint8Array(CONFIG.cache.L2CacheLines);
        // Amount of words (graphical slots) in one cacheline
        this.lineSize = CONFIG.cache.L2CacheLineSize;
        // Device memory access instructions waiting to return
        // TODO simulate throughtput limit from bus width here
        this.memoryAccessQueue = new Array();
    }

    align(i) {
        return i - i % this.lineSize;
    }

    step() {
        // Age all cache lines
        for (let i = 0; i < this.ages.length; ++i) {
            ++this.ages[i];
        }
        // Add all completed memory fetches as cache lines
        this.memoryAccessQueue.forEach(instruction => {
            if (instruction.isDone()) {
                const memoryIndex = instruction.data.index;
                const lineIndex = this.getCachedIndex(memoryIndex);
                if (lineIndex < 0) {
                    this.addNew(memoryIndex);
                } else {
                    this.ages[lineIndex] = 0;
                }
            }
        })
        // Delete all completed memory access instructions
        this.memoryAccessQueue = this.memoryAccessQueue.filter(instruction => {
            return !instruction.isDone();
        });
    }

    getCachedIndex(i) {
        const aligned = this.align(i) + 1;
        return this.lines.findIndex(cached => cached > 0 && aligned === cached);
    }

    getQueuedInstruction(i) {
        return this.memoryAccessQueue.find(instr => instr.data.index === i);
    }

    addLine(i, j) {
        // Add new cacheline i with cached index j
        this.lines[i] = this.align(j) + 1;
        this.ages[i] = 0;
    }

    // Replace the oldest cacheline with i
    addNew(i) {
        const oldestIndex = this.lines.reduce((oldest, _, index) => {
            return this.ages[index] > this.ages[oldest] ? index : oldest;
        }, 0);
        this.addLine(oldestIndex, i);
    }

    queueMemoryAccess(i) {
        let instruction = this.getQueuedInstruction(i);
        if (typeof instruction !== "undefined") {
            // Memory access at index i already queued
            return instruction;
        }
        // Create new instruction to access memory at index i
        instruction = Instruction.deviceMemoryAccess(i);
        this.memoryAccessQueue.push(instruction);
        return instruction;
    }

    // Simulate a memory access through L2, return true if the index was in the cache.
    // Also update the LRU age of the line i belongs to.
    fetch(i) {
        let fetchInstruction;
        const j = this.getCachedIndex(i);
        if (j < 0) {
            // i was not cached, queue fetch from memory
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
            return "cached";
        } else if (typeof this.getQueuedInstruction(i) !== "undefined") {
            return "pendingMemoryAccess";
        } else {
            return "notInCache";
        }
    }
}

// Namespace object for kernel simulation
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

    // Identity function with 1 cycle latency
    identity(x) {
        this.prevInstruction = new Instruction("identity", 1);
        return x;
    }

    // Simulated arithmetic instruction
    arithmetic(result) {
        // Create latency
        this.prevInstruction = Instruction.arithmetic();
        return result;
    }

    // Simulated random access memory transaction
    arrayGet(memoryGetHandle, index) {
        // Simulate memory get
        this.prevInstruction = memoryGetHandle(index);
        // Get the actual value without simulation
        return memoryGetHandle(index, true);
    }

    // Simulated random access memory transaction
    arraySet(memorySetHandle, index, value) {
        // Simulate memory set
        this.prevInstruction = memorySetHandle(index, false, value);
        // Set the actual value without simulation
        memorySetHandle(index, true, value);
    }

    // Jump from current line by offset
    jump(offset) {
        const instr = new Instruction("jump", 1);
        instr.data = {jumpOffset: offset};
        this.prevInstruction = instr;
    }
}

// Drawable, simulated area of GPU DRAM.
// Several DeviceMemory instances can be defined, e.g. for representing an input and output array
class DeviceMemory extends Drawable {
    constructor(x, y, width, height, canvas) {
        super(x, y, width, height, canvas);
        const rowSlotCount = CONFIG.memory.rowSlotCount;
        const columnSlotCount = CONFIG.memory.columnSlotCount;
        const slotSize = CONFIG.memory.slotSize;
        const slotPadding = CONFIG.memory.slotPadding;
        const slotFillRGBA = CONFIG.memory.slotFillRGBA;
        this.slots = Array.from(
            new Array(rowSlotCount * columnSlotCount),
            (_, i) => {
                const slotX = x + (i % rowSlotCount) * (slotSize + slotPadding);
                const slotY = y + Math.floor(i / rowSlotCount) * (slotSize + slotPadding);
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
}

// One memory slot represents a single address in RAM that holds a single word
class MemorySlot extends Drawable {
    constructor(index, value, ...drawableArgs) {
        super(...drawableArgs);
        this.index = index;
        this.value = value;
        this.hotness = 0;
        // Copy default color
        this.defaultColor = this.fillRGBA.slice();
        this.coolDownPeriod = CONFIG.memory.coolDownPeriod;
        this.coolDownStep = (1.0 - this.defaultColor[3]) / this.coolDownPeriod;
        this.cachedColor = CONFIG.cache.cachedStateRGBA.slice();
        this.pendingColor = CONFIG.cache.pendingStateRGBA.slice();
    }

    // Simulate a memory access to this index
    touch() {
        this.hotness = this.coolDownPeriod;
        this.fillRGBA[3] = 1.0;
    }

    // Update slot cache status to highlight cached slots in rendering
    setCachedState(state) {
        let newColor;
        switch(state) {
            case "cached":
                newColor = this.cachedColor;
                break;
            case "pendingMemoryAccess":
                newColor = this.pendingColor;
                break;
            case "notInCache":
            default:
                newColor= this.defaultColor;
                break;
        }
        // Don't update alpha if this slot is cooling down from a previous touch
        // if (this.hotness > 0 && this.fillRGBA[3] > newColor[3]) {
            // for (let i = 0; i < 3; ++i) {
                // this.fillRGBA[i] = newColor[i];
            // }
        if (this.hotness === 0) {
            this.fillRGBA = newColor.slice();
        }
    }

    // Animation step
    step() {
        this.draw();
        if (this.hotness > 0) {
            if (--this.hotness === 0) {
                this.fillRGBA = this.defaultColor.slice();
            } else {
                this.fillRGBA[3] -= this.coolDownStep;
            }
        }
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
        let threadIndexes = new Array;
        for (let j = 0; j < this.dim.y; ++j) {
            for (let i = 0; i < this.dim.x; ++i) {
                threadIndexes.push({x: i, y: j});
                if (threadIndexes.length === warpSize) {
                    const warp = new Warp(this, threadIndexes.slice(), kernelArgs);
                    threadIndexes = new Array;
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
            this.statement.apply(this.kernelContext);
            this.instruction = this.kernelContext.prevInstruction;
            this.statement = null;
        } else {
            // Continue waiting for instruction to complete
            this.instruction.cycle();
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
        // inelegant jump instruction hack, assuming all threads have the same jump instruction at the same cycle
        const instr = this.threads[0].instruction;
        if (instr !== null && instr.name === "jump" && !instr.isDone()) {
            this.programCounter += instr.data.jumpOffset;
        }
    }
}

// Simulated latency after an SM issued instruction
class Instruction {
    constructor(name, latency) {
        this.name = name;
        this.cyclesLeft = latency;
        this.data = {};
    }

    isDone() {
        return this.cyclesLeft === 0;
    }

    cycle() {
        if (this.cyclesLeft > 0) {
            --this.cyclesLeft;
        }
    }

    static empty() {
        return new Instruction("empty", 0);
    }

    static arithmetic() {
        return new Instruction("arithmetic", CONFIG.latencies.arithmetic);
    }

    static cachedMemoryAccess() {
        return new Instruction("cachedMemoryAccess", CONFIG.latencies.L2CacheAccess);
    }

    static deviceMemoryAccess(i) {
        const instr = new Instruction("deviceMemoryAccess", CONFIG.latencies.memoryAccess);
        instr.data = {index: i};
        return instr;
    }
}

class SMstats {
    constructor(stateElement) {
        this.stateElement = stateElement;
        this.cycleCounter = stateElement.querySelector("li pre span.sm-cycle-counter");
        this.cycles = 0;
    }

    cycle() {
        ++this.cycles;
        this.cycleCounter.innerHTML = this.cycles.toString();
    }
}

class SMController {
    constructor(id) {
        this.schedulerCount = CONFIG.SM.warpSchedulers;
        this.residentWarps = new Array();
        this.grid = null;
        this.program = null;
        this.activeBlock = null;
        this.kernelArgs = null;
        const stateElement = document.getElementById("sm-state-" + id);
        this.statsWidget = new SMstats(stateElement);
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

    // Return a generator of all free warps, available for scheduling
    *freeWarps() {
        for (let warp of this.nonTerminatedWarps())
            if (warp.isActive() && !warp.running)
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

    // Replace warps waiting for instructions to complete with active warps
    scheduleWarps() {
        const activeWarpCount = Array.from(this.freeWarps()).length;
        let scheduledWarpCount = Array.from(this.scheduledWarps()).length;
        let remainingWaiting = Math.min(this.schedulerCount, activeWarpCount);
        while (remainingWaiting-- > 0) {
            // Schedule first free warp
            for (let warp of this.freeWarps()) {
                // Schedule warp for execution in SM
                warp.running = true;
                ++scheduledWarpCount;
                break;
            }
            // If too many warps are executing, remove one
            if (scheduledWarpCount > this.schedulerCount) {
                for (let warp of this.blockingWarps()) {
                    warp.running = false;
                    --scheduledWarpCount;
                    break;
                }
            }
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
        this.updateProgramCounters();
        const nonTerminatedWarps = Array.from(this.nonTerminatedWarps());
        if (nonTerminatedWarps.length > 0) {
            // Warps available for execution
            nonTerminatedWarps.forEach(warp => warp.cycle());
        } else {
            // All warps have terminated, try to take next block from grid
            this.releaseProcessedBlock();
            const hasWork = this.scheduleNextBlock();
            if (!hasWork) {
                this.program = null;
            }
        }
    }
}

class StreamingMultiprocessor {
    constructor(id) {
        this.frameCounter = 0;
        this.framesPerCycle = CONFIG.SM.framesPerSMCycle;
        assert(this.framesPerCycle > 0, "frames per SM cycle must be at least 1");
        this.controller = new SMController(id);
    }

    // Simulate one processor cycle
    cycle() {
        this.frameCounter = 0;
        if (this.controller.program !== null) {
            this.controller.cycle();
        }
    }

    // Animation loop step
    step() {
        if (Math.random() < 0.1) {
            // Simulated latency within this SM for the duration of a single animation frame
            return;
        }
        if (++this.frameCounter === this.framesPerCycle) {
            this.cycle();
        }
    }
}

// Wrapper around the device memory and multiprocessors, simulating memory access handling and scheduling
class Device {
    constructor(memoryCanvas) {
        this.memory = new DeviceMemory(0, 0, memoryCanvas.width, memoryCanvas.height, memoryCanvas);
        this.multiprocessors = this.createProcessors(CONFIG.SM.count);
        this.kernelSource = null;
        this.L2Cache = new L2Cache();
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
        this.L2Cache.step();
        const L2CacheStateHandle = this.L2Cache.getCacheState.bind(this.L2Cache);
        this.memory.step(L2CacheStateHandle);
        this.multiprocessors.forEach((sm, smIndex) => {
            sm.step();
            sm.controller.residentWarps.forEach((warp, warpIndex) => {
                const colorIndex = {x: warpIndex, y: smIndex};
                const lineno = warp.programCounter;
                this.kernelSource.setHighlight(colorIndex, lineno, true);
            });
        });
        this.kernelSource.step();
    }
}

class KernelSource {
    constructor(sourceLines, lineHeight) {
        const palette = CONFIG.animation.kernelHighlightPalette;
        this.highlightedLines = Array.from(sourceLines, (line, lineno) => {
            const _ = undefined;
            const x = 0;
            const y = lineno * lineHeight;
            const width = kernelCanvas.width;
            return Array.from(palette, shades => {
                return Array.from(shades, highlightColor => {
                    const hlDrawable = new Drawable(x, y, width, lineHeight, kernelCanvas, _, highlightColor);
                    return {
                        drawable: hlDrawable,
                        on: false,
                    };
                });
            });
        });
    }

    setHighlight(colorIndex, lineno, on) {
        this.highlightedLines[lineno][colorIndex.y][colorIndex.x].on = on;
    }

    drawHighlighted() {
        this.highlightedLines.forEach(line => {
            line.forEach(shades => {
                shades.forEach(hl => {
                    if (hl.on) {
                        hl.drawable.draw();
                    }
                    hl.on = false;
                });
            });
        });
    }

    step() {
        this.drawHighlighted();
    }
}
