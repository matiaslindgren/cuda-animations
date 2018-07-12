"use strict";

// All mutable global variables
var memoryCanvas;
var SMCanvas;
var kernelCanvas;
var device;
var prevRenderTime = performance.now();
var drawing = true;

const kernelSourceLines = [
    "__global__ void kernel(const float* input) {",
    "    const int i = threadIdx.x + blockIdx.x * blockDim.x;",
    "    const float x = input[i];",
    "    float a = x * x;",
    "    float b = 2.0 * a;",
    "    float c = a * b + x;",
    "    float d = a + b + c;",
    "}",
];
const kernelCallableStatements = [
    function(input, n) {
        this.arithmetic();
        this.locals.i = this.threadIdx.x + this.blockIdx.x * this.blockDim.x;
    },
    function(input, n) {
        this.locals.x = this.arrayGet(input, this.locals.i);
    },
    function(input, n) {
        this.arithmetic();
        this.locals.a = this.locals.x * this.locals.x;
    },
    function(input, n) {
        this.arithmetic();
        this.locals.b = 2.0 * this.locals.a;
    },
    function(input, n) {
        this.arithmetic();
        this.locals.c = this.locals.a * this.locals.b + this.locals.x;
    },
    function(input, n) {
        this.arithmetic();
        this.locals.d = this.locals.a + this.locals.b + this.locals.c;
    },
];

function parseStylePX(style, prop) {
    return parseInt(style.getPropertyValue(prop).split("px")[0]);
}

function init() {
    // Initialize canvas element dimensions from computed stylesheet
    memoryCanvas = document.getElementById("memoryCanvas");
    SMCanvas = document.getElementById("SMCanvas");
    kernelCanvas = document.getElementById("kernelCanvas");
    [memoryCanvas, SMCanvas, kernelCanvas].forEach(canvas => {
        const style = window.getComputedStyle(canvas, null);
        canvas.width = parseStylePX(style, "width");
        canvas.height = parseStylePX(style, "height");
    });

    device = new Device();
    const grid = new Grid(CONFIG.grid.dimGrid, CONFIG.grid.dimBlock);
    const program = {
        sourceLines: kernelSourceLines,
        statements: kernelCallableStatements,
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
    clear(kernelCanvas);
    device.step();
    if (device.programTerminated()) {
        pause();
    }
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
