"use strict";

// All globals vars
var memoryCanvas;
var SMCanvas;
var memorySlots;
var config;

function init() {
    memoryCanvas = document.getElementById("memoryCanvas");
    SMCanvas = document.getElementById("SMCanvas");
    memorySlots = createGrid();
    config = {
        // The amount of animation render frames simulating one streaming multiprocessor cycle
        FramesPerSMCycle: 120
        memory: {
            rowSlotCount: 30,
            columnSlotCount: 30,
            slotPadding: 2,
            slotSize: 10,
            slotFillStyle: "#eee"
        }
    };
    // config.memory.slotSize = Math.floor(Math.min(
        // memoryCanvas.width/config.memory.rowSlotCount - config.memory.slotPadding,
        // memoryCanvas.height/config.memory.columnSlotCount - config.memory.slotPadding
    // ));

}

function clear(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = 'rgb(255, 255, 255)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawRects(rects, canvas) {
    const ctx = canvas.getContext("2d");
    rects.forEach(rect => {
        if (typeof rect.fillStyle !== "undefined") {
            ctx.fillStyle = rect.fillStyle;
            ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        }
        if (typeof rect.strokeStyle !== "undefined") {
            ctx.strokeStyle = rect.strokeStyle;
            ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
        }
    });
}

class DeviceMemory {
    constructor() {
        this.config = config.memory;
    }

    createGrid() {
        const rowSlotCount = this.config.rowSlotCount;
        const columnSlotCount = this.config.columnSlotCount;
        const slotSize = this.config.slotSize;
        const slotPadding = this.config.slotPadding;
        return Array.from(
            new Array(rowSlotCount * columnSlotCount),
            (_, i) => {
                const x = (i % rowSlotCount) * (slotSize + slotPadding);
                const y = Math.floor(i / rowSlotCount) * (slotSize + slotPadding);
                return new MemorySlot(x, y, slotSize);
            }
        );
    }

}

class MemorySlot {
    constructor(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.fillStyle = config.memory.slotFillStyle;
    }

    // Request this index in memory
    touch() {
    }

    // Animation step
    step() {
    }

}

class Thread {
}

class Warp {
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
        if (this.frameCounter <= config.FramesPerSMCycle) {
            ++this.frameCounter;
        } else if (this.frameCounter === config.FramesPerSMCycle) {
            this.doCycle();
            this.frameCounter = 0;
        } else {
            console.error("ERROR: inconsistent SM counter state, exceeded FramesPerSMCycle");
        }
    }
}

function step() {
    processor.forEach((sm, i) => {
        sm.step();
    });
}

function draw(now) {
    clear(memoryCanvas);
    clear(SMCanvas);
    drawRects(slots, memoryCanvas);
    processors.step();
    step(memory);
    window.requestAnimationFrame(draw);
}

init();
window.requestAnimationFrame(draw);
