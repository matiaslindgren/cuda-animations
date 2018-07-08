"use strict";

// All mutable global variables
var memoryCanvas;
var SMCanvas;
var device;
var prevRenderTime = performance.now();
var drawing = true;

function init() {
    memoryCanvas = document.getElementById("memoryCanvas");
    SMCanvas = document.getElementById("SMCanvas");
    device = new Device();
    const grid = new Grid(CONFIG.grid.dimGrid, CONFIG.grid.dimBlock);
    const program = {
        source: "__global__ void kernel(const float* input) {\n    input[threadIdx.x];\n}",
        statements: [
            function(input, n) {
                this.locals.i = this.threadIdx.x + this.blockIdx.x * this.blockDim.x;
                this.locals.x = this.arrayGet(input, this.locals.i);
            },
            function(input, n) {
                this.locals.i = this.threadIdx.x + this.blockIdx.x * this.blockDim.x;
                this.locals.x = this.arrayGet(input, this.locals.i);
            },
        ],
    };
    device.setProgram(grid, program);
}

function clear(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function draw(now) {
    if (drawing) {
        window.requestAnimationFrame(draw);
    }
    if (!drawing || now - prevRenderTime < CONFIG.animation.drawDelayMS) {
        // Throttle rendering speed to avoid choking the CPU
        return;
    }
    prevRenderTime = now;
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
    drawing = !drawing;
    if (drawing) {
        window.requestAnimationFrame(draw);
    }
}

document.addEventListener("DOMContentLoaded", restart);
