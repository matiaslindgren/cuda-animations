"use strict";

const config = {
    // The amount of animation render frames simulating one streaming multiprocessor cycle
    framesPerSMCycle: 120,
    memory: {
        // Amount of indexable memory slots on each row and column
        rowSlotCount: 30,
        columnSlotCount: 30,
        // Empty space between each slot on all sides
        slotPadding: 2,
        slotSize: 18,
        slotFillRGBA: [100, 100, 100, 0.2],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 30,
    }
};

class Drawable {
    constructor(x, y, width, height, canvas, strokeRGBA, fillRGBA) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.canvasContext = canvas.getContext("2d");
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

class DeviceMemory extends Drawable {
    constructor(x, y, width, height, canvas) {
        super(x, y, width, height, canvas);
        const rowSlotCount = config.memory.rowSlotCount;
        const columnSlotCount = config.memory.columnSlotCount;
        const slotSize = config.memory.slotSize;
        const slotPadding = config.memory.slotPadding;
        const slotFillRGBA = config.memory.slotFillRGBA;
        this.memorySlots = Array.from(
            new Array(rowSlotCount * columnSlotCount),
            (_, i) => {
                const slotX = (i % rowSlotCount) * (slotSize + slotPadding);
                const slotY = Math.floor(i / rowSlotCount) * (slotSize + slotPadding);
                return new MemorySlot(i, x + slotX, y + slotY, slotSize, slotSize, canvas, undefined, slotFillRGBA);
            }
        );
    }

    step() {
        super.draw();
        this.memorySlots.forEach(slot => slot.step());
    }
}

class MemorySlot extends Drawable {
    constructor(index, ...drawableArgs) {
        super(...drawableArgs);
        this.index = index;
        this.hotness = 0;
        // Copy default color
        this.coolColor = this.fillRGBA.slice();
        this.coolDownPeriod = config.memory.coolDownPeriod;
        this.coolDownStep = (1.0 - this.coolColor[3]) / this.coolDownPeriod;
    }

    // Simulate a memory access to this index
    touch() {
        this.hotness = this.coolDownPeriod;
        this.fillRGBA[3] = 1.0;
    }

    // Animation step
    step() {
        this.draw();
        if (this.hotness > 0) {
            --this.hotness;
            if (this.hotness === 0) {
                this.fillRGBA = this.coolColor;
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

class StreamingMultiprocessor {
    constructor() {
        this.frameCounter = 0;
        this.cycleCounter = 0;
        this.warps = [];
    }

    // One simulated processor cycle
    doCycle() {
        ++this.cycleCounter;
    }

    // Animation loop step
    step() {
        if (this.frameCounter <= config.framesPerSMCycle) {
            ++this.frameCounter;
        } else if (this.frameCounter === config.framesPerSMCycle) {
            this.doCycle();
            this.frameCounter = 0;
        } else {
            console.error("ERROR: inconsistent SM counter state, exceeded framesPerSMCycle");
        }
    }
}

var memoryCanvas;
var SMCanvas;
var deviceMemory;
var multiprocessors;

function init() {
    memoryCanvas = document.getElementById("memoryCanvas");
    SMCanvas = document.getElementById("SMCanvas");
    deviceMemory = new DeviceMemory(0, 0, memoryCanvas.width, memoryCanvas.height, memoryCanvas);
}

function clear(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Throttle FPS to avoid choking the CPU
var then = performance.now();
const drawDelayMS = 1000.0 / 30.0;

function draw(now) {
    window.requestAnimationFrame(draw);
    if (now - then < drawDelayMS) {
        return;
    }
    then = now;
    clear(memoryCanvas);
    clear(SMCanvas);
    deviceMemory.step();
    if (Math.random() < 0.1) {
        const i = Math.floor(Math.random() * deviceMemory.memorySlots.length);
        deviceMemory.memorySlots[i].touch();
    }
}

init();
window.requestAnimationFrame(draw);
