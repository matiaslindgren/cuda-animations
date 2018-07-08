"use strict";

function assert(expr, msg) {
    if (!expr)
        console.error("ASSERTION FAILED", msg);
}

const CONFIG = {
    animation: {
        // Delay in ms between rendering each frame
        drawDelayMS: 1000.0 / 30.0,
    },
    memory: {
        // Amount of indexable memory slots on each row and column
        rowSlotCount: 32,
        columnSlotCount: 32,
        // Empty space between each slot on all sides
        slotPadding: 2,
        slotSize: 23,
        slotFillRGBA: [100, 100, 100, 0.15],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 10,
    },
    cache: {
        // Size of a L2 cacheline in words
        L2CacheLineSize: 8,
        L2CacheLines: 16,
        cachedStateRGBA: [100, 100, 220, 0.5],
    },
    SM: {
        count: 4,
        warpSize: 32,
        warpSchedulers: 2,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 1,
        paddingX: 20,
        paddingY: 20,
        height: 135,
        fillRGBA: [100, 100, 100, 0.1],
    },
    grid: {
        dimGrid: {x: Math.round(32*32/128)},
        dimBlock: {x: 128},
    },
};

class Drawable {
    constructor(x, y, width, height, canvas, strokeRGBA, fillRGBA, text) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.canvasContext = canvas.getContext("2d");
        // Copy RGBA arrays, if given, else set to null
        this.strokeRGBA = (typeof strokeRGBA === "undefined") ? null : strokeRGBA.slice();
        this.fillRGBA = (typeof fillRGBA === "undefined") ? null : fillRGBA.slice();
        // If text is given, this drawable renders text
        this.drawableText = (typeof text === "undefined") ? null : text;
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
        if (this.drawableText !== null) {
            const fontsize = 25;
            ctx.font = fontsize + "px monospace";
            ctx.fillStyle = "rgba(0, 0, 0, 1)";
            ctx.fillText(this.drawableText, x + fontsize, y + fontsize);
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
        // Amount of words in one cacheline
        this.lineSize = CONFIG.cache.L2CacheLineSize;
        // Device memory indexes currently in cache,
        // for constant time lookup during rendering (a cache of a simulated cache...).
        this.cachedIndexes = new Set();
    }

    align(i) {
        return i - i % this.lineSize;
    }

    step() {
        for (let i = 0; i < this.ages.length; ++i) {
            ++this.ages[i];
        }
    }

    getFromCache(i) {
        const aligned = this.align(i) + 1;
        return this.lines.findIndex(cached => cached > 0 && aligned === cached);
    }

    clearLine(i) {
        // Delete all device memory indexes from lookup set for this cacheline
        const lineStart = this.lines[i] - 1;
        if (lineStart < 0) {
            // Cacheline already empty
            return;
        }
        for (let j = 0; j < this.lineSize; ++j) {
            this.cachedIndexes.delete(j + lineStart);
        }
    }

    addLine(i, j) {
        // Add new cacheline i with cached index j
        this.lines[i] = j;
        this.ages[i] = 0;
        // Update lookup set
        const lineStart = this.lines[i] - 1;
        for (let j = 0; j < this.lineSize; ++j) {
            this.cachedIndexes.add(j + lineStart);
        }
    }

    // Replace the oldest cacheline with i
    addNew(i) {
        const oldestIndex = this.lines.reduce((oldest, _, index) => {
            return this.ages[index] > this.ages[oldest] ? index : oldest;
        }, 0);
        this.clearLine(oldestIndex);
        this.addLine(oldestIndex, this.align(i) + 1);
    }

    // Simulate a memory access through L2, return true if the index was in the cache.
    // Also update the LRU age of the line i belongs to.
    fetch(i) {
        this.step();
        const j = this.getFromCache(i);
        if (j < 0) {
            // i was not cached, replace oldest cacheline
            this.addNew(i);
            return false;
        } else {
            // i was cached, set age of i's cacheline to zero
            this.ages[j] = 0;
            return true;
        }
    }

    // Check if device memory index i is in some of the cache lines
    isCached(i) {
        return this.cachedIndexes.has(i);
    }
}

class DeviceMemory extends Drawable {
    constructor(x, y, width, height, canvas) {
        super(x, y, width, height, canvas);
        const rowSlotCount = CONFIG.memory.rowSlotCount;
        const columnSlotCount = CONFIG.memory.columnSlotCount;
        const slotSize = CONFIG.memory.slotSize;
        const slotPadding = CONFIG.memory.slotPadding;
        const slotFillRGBA = CONFIG.memory.slotFillRGBA;
        this.L2Cache = new L2Cache();
        this.slots = Array.from(
            new Array(rowSlotCount * columnSlotCount),
            (_, i) => {
                const slotX = x + (i % rowSlotCount) * (slotSize + slotPadding);
                const slotY = y + Math.floor(i / rowSlotCount) * (slotSize + slotPadding);
                return new MemorySlot(i, slotX, slotY, slotSize, slotSize, canvas, undefined, slotFillRGBA);
            }
        );
    }

    // Simulate a memory access to index i in the global memory
    access(i) {
        this.slots[i].touch();
        if (this.L2Cache.fetch(i)) {
            // i was in a cacheline
            return Instruction.cachedMemoryAccess();
        } else {
            // i was not in a cacheline
            return Instruction.deviceMemoryAccess();
        }
    }

    step() {
        this.slots.forEach((slot, i) => {
            slot.setCachedState(this.L2Cache.isCached(i));
            slot.step();
        });
        super.draw();
    }
}

// One memory slot represents a single address in RAM that holds a single word
class MemorySlot extends Drawable {
    constructor(index, ...drawableArgs) {
        super(...drawableArgs);
        this.index = index;
        this.hotness = 0;
        this.cached = false;
        // Copy default color
        this.defaultColor = this.fillRGBA.slice();
        this.coolDownPeriod = CONFIG.memory.coolDownPeriod;
        this.coolDownStep = (1.0 - this.defaultColor[3]) / this.coolDownPeriod;
        this.cachedColor = CONFIG.cache.cachedStateRGBA.slice();
    }

    // Simulate a memory access to this index
    touch() {
        this.hotness = this.coolDownPeriod;
        this.fillRGBA[3] = 1.0;
    }

    // Update slot cache status to highlight cached slots in rendering
    setCachedState(isCached) {
        this.cached = isCached;
        const newColor = (isCached ? this.cachedColor : this.defaultColor).slice();
        // Don't update alpha if this slot is cooling down from a previous touch
        if (this.hotness > 0 && this.fillRGBA[3] > newColor[3]) {
            for (let i = 0; i < 3; ++i) {
                this.fillRGBA[i] = newColor[i];
            }
        } else {
            this.fillRGBA = newColor;
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
            new Array(dimGrid.x),
            (_, i) => new Block(dimBlock)
        );
    }

    nextFreeBlockIndex() {
        return this.blocks.findIndex(block => !block.locked && !block.processed);
    }
}

// Block of threads
class Block {
    constructor(dimBlock) {
        this.locked = false;
        this.processed = false;
        this.dim = dimBlock;
    }

    // Generator of thread warps from this block,
    // takes as argument the memory access handle of the SM controller this block is being assigned to
    *asWarps(memoryAccessHandle) {
        const warpSize = CONFIG.SM.warpSize;
        let warpThreads = new Array();
        for (let i = 0; i < this.dim.x; ++i) {
            warpThreads.push(new Thread(i, memoryAccessHandle));
            if (warpThreads.length === warpSize) {
                yield new Warp(warpThreads.slice());
                warpThreads = new Array();
            }
        }
        if (warpThreads.length > 0) {
            // If block size is not divisible by warp size, fill warp with trailing dummy threads
            while (warpThreads.length < warpSize) {
                warpThreads.push(new Thread(-1), _ => console.error("WARNING, -1 threads should not try to access memory"));
            }
            yield new Warp(warpThreads.slice());
        }
    }
}

// Simulated CUDA thread
class Thread {
    constructor(x, memoryAccessHandle) {
        this.x = x;
        this.statement = null;
        this.instruction = Instruction.empty();
        this.memoryAccessHandle = memoryAccessHandle;
    }

    // True if this thread is ready to take a new instruction
    isActive() {
        return this.instruction.isDone();
    }

    access(i) {
        return this.memoryAccessHandle(i);
    }

    cycle() {
        if (this.x < 0)
            return;
        if (this.statement !== null) {
            this.statement.forEach(expr => expr.setContext(this));
            this.instruction = CUDAExpression.evalNested(this.statement);
            this.statement = null;
        }
        this.instruction.cycle();
    }
}

// Simulated group of CUDA threads running in an SM
class Warp {
    constructor(threads) {
        this.terminated = false;
        this.running = false;
        this.threads = threads;
        // Assuming pre-Volta architecture
        // Since Volta, each thread has its own PC
        this.programCounter = 0;
    }

    // An active warp is ready to execute the next instruction
    isActive() {
        return this.threads.every(t => t.isActive());
    }

    nextStatement(statement) {
        this.threads.forEach(t => t.statement = statement);
        ++this.programCounter;
    }

    cycle() {
        this.threads.forEach(t => t.cycle());
    }
}

// Simulated latency after an SM issued instruction
class Instruction {
    constructor(latency) {
        this.cyclesLeft = latency;
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
        return new Instruction(0);
    }

    static arithmetic() {
        return new Instruction(6);
    }

    static cachedMemoryAccess() {
        return new Instruction(100);
    }

    static deviceMemoryAccess() {
        return new Instruction(360);
    }
}

class CUDAExpression {
    constructor(callable, type) {
        this.callable = callable;
        this.type = type;
        this.context = null;
    }

    setContext(c) {
        this.context = c;
    }

    eval(arg) {
        if (this.context === null) {
            console.error("ERROR: trying to evaluate CUDAExpression of type: " + this.type + " without context, aka 'what is this'");
            return;
        }
        return this.callable.call(this.context, arg);
    }

    static evalNested(expressions) {
        const r = expressions.reduce((result, expr) => expr.eval(result), undefined);
        console.log("evalnested to ", r);
        return r;
    }

    static threadIdx_x() {
        return new CUDAExpression(function() { return this.x; }, "threadIdx.x");
    }

    static memoryAccess() {
        return new CUDAExpression(function(i) { console.log("returning memory access from", this); return this.access(i); }, "memoryAccess");
    }
}

// Drawable counter of SM cycles
class CycleCounter extends Drawable {
    constructor(...drawableArgs) {
        super(...drawableArgs, '0');
        this.cycles = 0;
    }

    cycle() {
        ++this.cycles;
        this.drawableText = "cycle: " + this.cycles.toString();
    }

    step() {
        super.draw();
    }
}

class SMController {
    constructor() {
        this.schedulerCount = CONFIG.SM.warpSchedulers;
        this.residentWarps = new Array();
        this.grid = null;
        this.program = null;
        this.activeBlock = null;
        this.memoryAccessHandle = null;
    }

    // Take next unlocked, unprocessed block from the grid
    takeBlock() {
        const i = this.grid.nextFreeBlockIndex();
        if (i < 0) {
            // No free blocks
            return false;
        }
        const newBlock = this.grid.blocks[i];
        this.activeBlock = newBlock;
        this.activeBlock.locked = true;
        this.residentWarps = Array.from(this.activeBlock.asWarps(this.memoryAccessHandle));
        return true;
    }

    releaseActiveBlock() {
        if (this.activeBlock !== null) {
            this.activeBlock.locked = false;
            this.activeBlock = null;
        }
    }

    // Return a generator of all non-terminated warps
    *runningWarps() {
        for (let warp of this.residentWarps)
            if (!warp.terminated)
                yield warp;
    }

    // Return a generator of all free warps, available for scheduling
    *freeWarps() {
        for (let warp of this.runningWarps())
            if (warp.isActive() && !warp.running)
                yield warp;
    }

    // Return a generator of all scheduled warps
    *scheduledWarps() {
        for (let warp of this.runningWarps())
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
        const x = Array.from(this.scheduledWarps()).length;
        assert(x === this.schedulerCount, "invalid amount of scheduled warps " + x);
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
        this.scheduleWarps();
        this.updateProgramCounters();
        const runningWarps = Array.from(this.runningWarps());
        if (runningWarps.length > 0) {
            // Warps available for execution
            runningWarps.forEach(warp => warp.cycle());
        } else {
            this.releaseActiveBlock();
            // All warps have terminated, try to take next block from grid
            if (this.takeBlock()) {
                // No free blocks, grid fully processed
                this.program = null;
            }
        }
    }
}

class StreamingMultiprocessor extends Drawable {
    constructor(...drawableArgs) {
        super(...drawableArgs);
        this.frameCounter = 0;
        this.cycleLabel = new CycleCounter(...drawableArgs);
        this.framesPerCycle = CONFIG.SM.framesPerSMCycle;
        assert(this.framesPerCycle > 0, "frames per SM cycle must be at least 1");
        this.controller = new SMController();
    }

    // Simulate one processor cycle
    cycle() {
        this.frameCounter = 0;
        if (this.controller.program !== null) {
            this.cycleLabel.cycle();
            this.controller.cycle();
        }
    }

    // Animation loop step
    step() {
        super.draw();
        this.cycleLabel.step();
        if (Math.random() < 0.15) {
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
    constructor() {
        this.memory = new DeviceMemory(0, 0, memoryCanvas.width, memoryCanvas.height, memoryCanvas);
        this.multiprocessors = this.createProcessors(CONFIG.SM.count);
    }

    // Initialize all processors
    setProgram(grid, program) {
        const memoryAccessHandle = this.accessMemory.bind(this);
        this.multiprocessors.forEach(sm => {
            assert(sm.controller.program === null, "sm controllers should not be reset while they are running a program");
            sm.controller.memoryAccessHandle = memoryAccessHandle;
            sm.controller.program = program;
            sm.controller.grid = grid;
            sm.controller.takeBlock();
        });
    }

    createProcessors(count) {
        return Array.from(
            new Array(count),
            (_, i) => {
                const x = CONFIG.SM.paddingX;
                const y = i * CONFIG.SM.height + (i + 1) * CONFIG.SM.paddingY;
                const width = SMCanvas.width - 2 * CONFIG.SM.paddingX;
                const height = CONFIG.SM.height;
                return new StreamingMultiprocessor(x, y, width, height, SMCanvas, undefined, CONFIG.SM.fillRGBA);
            }
        );
    }

    accessMemory(i) {
        return this.memory.access(i);
    }

    step() {
        this.memory.step();
        this.multiprocessors.forEach(sm => sm.step());
    }
}
