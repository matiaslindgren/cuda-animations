"use strict";

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
        slotSize: 22,
        slotFillRGBA: [100, 100, 100, 0.15],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 10,
        // Size of a L2 cacheline in words
        L2CacheLineSize: 8,
        L2CacheLines: 8,
    },
    SM: {
        count: 4,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 1,
        paddingX: 20,
        paddingY: 20,
        height: 100,
        fillRGBA: [100, 100, 100, 0.3],
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
        this.lines = new Uint16Array(CONFIG.memory.L2CacheLines);
        this.ages = new Uint8Array(CONFIG.memory.L2CacheLines);
        // Amount of words in one cacheline
        this.lineSize = CONFIG.memory.L2CacheLineSize;
    }

    align(i) {
        return i - i % this.lineSize;
    }

    age() {
        for (let i = 0; i < this.ages.length; ++i) {
            ++this.ages[i];
        }
    }

    getFromCache(i) {
        const aligned = this.align(i) + 1;
        return this.lines.findIndex(cached => cached > 0 && aligned === cached);
    }

    // Replace the oldest cacheline with i
    addNew(i) {
        const oldestIndex = this.lines.reduce((oldest, _, index) => {
            return this.ages[index] > this.ages[oldest] ? index : oldest;
        }, 0);
        this.lines[oldestIndex] = this.align(i) + 1;
        this.ages[oldestIndex] = 0;
    }

    // Simulate a memory access through L2, return true if the index was in the cache.
    // Also update the LRU age of the line i belongs to.
    fetch(i) {
        this.age();
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

    // Return a generator of all cached device memory indexes in non-empty cachelines
    *cachedIndexes() {
        for (const lineStart of this.lines) {
            if (lineStart === 0) {
                continue;
            }
            for (let i = 0; i < this.lineSize; ++i) {
                yield lineStart + i - 1;
            }
        }
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
        for (const i of this.L2Cache.cachedIndexes()) {
            this.slots[i].setCached(true);
        }
        for (const slot of this.slots) {
            slot.step();
        }
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
        this.coolColor = this.fillRGBA.slice();
        this.coolDownPeriod = CONFIG.memory.coolDownPeriod;
        this.coolDownStep = (1.0 - this.coolColor[3]) / this.coolDownPeriod;
    }

    // Simulate a memory access to this index
    touch() {
        this.hotness = this.coolDownPeriod;
        this.fillRGBA[3] = 1.0;
    }

    // Update slot cache status to render it differently
    setCached(cached) {
        this.cached = cached;
        this.fillRGBA[1] = (cached) ? 200 : this.coolColor[1];
    }

    // Animation step
    step() {
        this.draw();
        if (this.hotness > 0) {
            --this.hotness;
            if (this.hotness === 0) {
                this.fillRGBA = this.coolColor.slice();
            } else {
                this.fillRGBA[3] -= this.coolDownStep;
            }
        }
    }
}

class Block {
}

class Thread {
}

class Warp {
}

class Instruction {
    constructor(latency) {
        this.latency = latency;
    }

    static arithmetic() {
        return new Instruction(6);
    }

    static deviceMemoryAccess() {
        return new Instruction(360);
    }

    static cachedMemoryAccess() {
        return new Instruction(160);
    }
}

class CycleCounter extends Drawable {
    constructor(...drawableArgs) {
        super(...drawableArgs, '0');
        this.cycles = 0;
    }

    doCycle() {
        ++this.cycles;
        this.drawableText = this.cycles.toString();
    }

    step() {
        super.draw();
    }
}

class StreamingMultiprocessor extends Drawable {
    constructor(...drawableArgs) {
        super(...drawableArgs);
        this.frameCounter = 0;
        this.cycleCounter = new CycleCounter(...drawableArgs);
        this.warps = [];
        this.framesPerCycle = CONFIG.SM.framesPerSMCycle;
        this.scheduler = null;
    }

    // Animation loop step
    step() {
        super.draw();
        this.cycleCounter.step();
        if (Math.random() < 0.15) {
            // Simulated latency within this SM for the duration of a single animation frame
            return;
        }
        if (this.frameCounter < this.framesPerCycle) {
            ++this.frameCounter;
            if (this.frameCounter === this.framesPerCycle) {
                // Simulate one processor cycle
                this.cycleCounter.doCycle();
                this.frameCounter = 0;
            }
        } else {
            console.error("ERROR: inconsistent SM counter state, exceeded framesPerSMCycle");
        }
    }
}

class Device {
    constructor() {
        this.memory = new DeviceMemory(0, 0, memoryCanvas.width, memoryCanvas.height, memoryCanvas);
        this.multiprocessors = Array.from(
            new Array(CONFIG.SM.count),
            (_, i) => {
                const x = CONFIG.SM.paddingX;
                const y = i * CONFIG.SM.height + (i + 1) * CONFIG.SM.paddingY;
                const width = SMCanvas.width - 2 * CONFIG.SM.paddingX;
                const height = CONFIG.SM.height;
                return new StreamingMultiprocessor(x, y, width, height, SMCanvas, undefined, CONFIG.SM.fillRGBA);
            }
        );
    }

    step() {
        this.memory.step();
        if (Math.random() < 0.05) {
            const i = Math.floor(Math.random() * 512);
            this.memory.access(i);
        }
        this.multiprocessors.forEach(sm => sm.step());
    }
}

///////////////////////////
// End of class definitions
///////////////////////////

var memoryCanvas;
var SMCanvas;
var device;

function init() {
    memoryCanvas = document.getElementById("memoryCanvas");
    SMCanvas = document.getElementById("SMCanvas");
    device = new Device();
}

function clear(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Throttle rendering speed to avoid choking the CPU
var then = performance.now();

var drawing = true;

function draw(now) {
    if (drawing) {
        window.requestAnimationFrame(draw);
    }
    if (!drawing || now - then < CONFIG.animation.drawDelayMS) {
        return;
    }
    then = now;
    clear(memoryCanvas);
    clear(SMCanvas);
    device.step();
}

function restart() {
    init();
    drawing = true;
    window.requestAnimationFrame(draw);
}

function pause() {
    if (!drawing) {
        window.requestAnimationFrame(draw);
    }
    drawing = !drawing;
}

restart();
