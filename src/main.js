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
        this.locals.i = this.arithmetic(this.threadIdx.x + this.blockIdx.x * this.blockDim.x);
    },
    function(input, n) {
        this.locals.x = this.arrayGet(input, this.locals.i);
    },
    function(input, n) {
        this.locals.a = this.arithmetic(this.locals.x * this.locals.x);
    },
    function(input, n) {
        this.locals.b = this.arithmetic(2.0 * this.locals.a);
    },
    function(input, n) {
        this.locals.c = this.arithmetic(this.locals.a * this.locals.b + this.locals.x);
    },
    function(input, n) {
        this.locals.d = this.arithmetic(this.locals.a + this.locals.b + this.locals.c);
    },
];

function parseStyle(style, prop, unit) {
    if (typeof unit === "undefined") {
        unit = "px";
    }
    return parseFloat(style.getPropertyValue(prop).split(unit)[0]);
}

function resetSizeAttrsFromStyle(element) {
    const style = window.getComputedStyle(element, null);
    element.width = parseStyle(style, "width");
    element.height = parseStyle(style, "height");
}

function resetSizeFromElement(source, target) {
    const style = window.getComputedStyle(source, null);
    target.width = Math.ceil(parseStyle(style, "width"));
    target.height = Math.ceil(parseStyle(style, "height"));
    target.style.width = target.width.toString() + "px";
    target.style.height = target.height.toString() + "px";
}

function init() {
    memoryCanvas = document.getElementById("memoryCanvas");
    SMCanvas = document.getElementById("SMCanvas");
    kernelCanvas = document.getElementById("kernelCanvas");

    // Initialize canvas element dimensions from computed stylesheet
    resetSizeAttrsFromStyle(memoryCanvas);
    resetSizeAttrsFromStyle(SMCanvas);

    // Render kernel source to set pre-element size
    const kernelSource = document.getElementById("kernelSource");
    kernelSource.innerText = kernelSourceLines.join('\n');
    // Align kernel source highlighting over the pre-element containing the source
    resetSizeFromElement(kernelSource, kernelCanvas);

    const sourceStyle = window.getComputedStyle(kernelSource, null);
    const sourceLineHeight = parseStyle(sourceStyle, "line-height", "em");

    // Initialize simulated GPU
    device = new Device();
    const grid = new Grid(CONFIG.grid.dimGrid, CONFIG.grid.dimBlock);
    const program = {
        sourceLines: kernelSourceLines,
        sourceLineHeight: sourceLineHeight,
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
