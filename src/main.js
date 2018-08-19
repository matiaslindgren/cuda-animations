"use strict";

// All mutable global variables
var memoryCanvasInput;
var memoryCanvasOutput;
var kernelCanvas;
var device;
var prevRenderTime = performance.now();
var drawing = true;

const kernelSourceLines = [
    "__global__ void kernel(float* output, const float* input, int n) {",
    "    const int i = threadIdx.x + blockIdx.x * blockDim.x;",
    "    const int j = threadIdx.y + blockIdx.y * blockDim.y;",
    "    float v = HUGE_VALF;",
    "    for (int k = 0; k < n; ++k) {",
    "        float x = input[n*i + k];",
    "        float y = input[n*k + j];",
    "        float z = x + y;",
    "        v = min(v, z);",
    "    }",
    "    output[n*i + j] = v;",
    "}",
];
const kernelCallableStatements = [
    function() {
        this.locals.i = this.arithmetic(this.threadIdx.x + this.blockIdx.x * this.blockDim.x);
    },
    function() {
        this.locals.j = this.arithmetic(this.threadIdx.y + this.blockIdx.y * this.blockDim.y);
    },
    function() {
        this.locals.v = this.identity(Infinity);
    },
    function() {
        this.locals.k = this.identity(0);
    },
    function() {
        this.locals.x = this.arrayGet(this.args.input, this.args.n * this.locals.i + this.locals.k);
    },
    function() {
        this.locals.y = this.arrayGet(this.args.input, this.args.n * this.locals.k + this.locals.j);
    },
    function() {
        this.locals.z = this.arithmetic(this.locals.x + this.locals.y);
    },
    function() {
        this.locals.v = this.arithmetic(Math.min(this.locals.v, this.locals.z));
    },
    function() {
        ++this.locals.k;
        if (false && this.locals.k < this.args.n) {
            this.jump(-4);
        }
    },
    function() {
        this.arraySet(this.args.output, this.args.n * this.locals.i + this.locals.j, this.locals.v);
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
    memoryCanvasInput = document.getElementById("memoryCanvasInput");
    memoryCanvasOutput = document.getElementById("memoryCanvasOutput");
    kernelCanvas = document.getElementById("kernelCanvas");

    // Initialize canvas element dimensions from computed stylesheet
    [memoryCanvasInput, memoryCanvasOutput].forEach(canvas => resetSizeAttrsFromStyle(canvas));

    // Render kernel source to set pre-element size
    const kernelSource = document.getElementById("kernelSource");
    kernelSource.innerText = kernelSourceLines.join('\n');
    // Align kernel source highlighting over the pre-element containing the source
    resetSizeFromElement(kernelSource, kernelCanvas);

    const sourceStyle = window.getComputedStyle(kernelSource, null);
    const sourceLineHeight = parseStyle(sourceStyle, "line-height", "em");

    // Initialize simulated GPU
    device = new Device(memoryCanvasInput);
    const grid = new Grid(CONFIG.grid.dimGrid, CONFIG.grid.dimBlock);
    const kernelArgs = {
        output: device.memoryTransaction.bind(device, "set"),
        input: device.memoryTransaction.bind(device, "get"),
        n: 32,
    };
    const program = {
        sourceLines: kernelSourceLines,
        sourceLineHeight: sourceLineHeight,
        statements: kernelCallableStatements,
        kernelArgs: kernelArgs,
    };
    if (kernelSourceLines.length - 2 !== kernelCallableStatements.length) {
        console.error("WARNING: Inconsistent kernel source line count when compared to callable statements");
    }
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
    clear(memoryCanvasInput);
    clear(memoryCanvasOutput);
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

// Define menubar buttons for interacting with the animation
function initUI() {
    const pauseButton = document.querySelector("#pause-button");
    const restartButton = document.querySelector("#restart-button");
    pauseButton.addEventListener("click", _ => {
        pause();
        pauseButton.value = drawing ? "Pause" : "Continue";
    });
    restartButton.addEventListener("click", _ => {
        pauseButton.value = "Pause";
        restart();
    });
}

document.addEventListener("DOMContentLoaded", _ => {
    initUI();
    restart();
});
