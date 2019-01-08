"use strict";

// All mutable global variables

var memoryCanvasInput;
var kernelCanvas;
var device;
var drawing = true;
var prevRenderFrameID;

// Application state, mutated with menu bar elements

// Choose default CUDA kernel from kernels.js
var activeKernel = "ppcStepV0";
// Amount of streaming multiprocessors in device
var smCount = 1;
// Amount of cache lines
var cacheLineCount = 0;
// Simulated instruction latencies as SM cycles
var instructionLatencies = "low";
// Should the kernel source lines be highlighted with the SM color or not
var highlightKernelLines = "on";
var memorySlotSize = 16;


function makeSMlistBody(count) {
    function liWrap(s, liID) {
        if (typeof liID === "undefined") {
            return "<li>" + s + "</li>";
        } else {
            return "<li id=\"sm-state-" + liID + "\">" + s + "</li>";
        }
    }

    function SMcontentsToUL(body) {
        return "<ul>" + Array.from(body, li => liWrap(li)).join("\n") + "</ul>";
    }

    const defaultSMstateBody = [
        '<pre>cycle <span class="sm-cycle-counter">0</span></pre>',
        '<pre>block <span class="sm-current-block-idx">&ltnone&gt</span></pre>',
    ];
    const liElements = Array.from(new Array(count), (_, i) => {
        return liWrap(SMcontentsToUL(defaultSMstateBody), i + 1);
    });
    return liElements.join("\n");
}

function makeKernelSelectOptionsHTML(kernels) {
    function makeOption(key) {
        return "<option value=\"" + key + "\">CUDA kernel: " + kernels[key].displayName + "</option>";
    }
    // Create all kernels as options HTML, where the default kernel is first
    const kernelsNoDefault = Object.keys(kernels).filter(k => k !== activeKernel);
    const optionsHTML = Array.from([activeKernel].concat(kernelsNoDefault), makeOption);
    return optionsHTML.join("\n");
}

function makeSMCountSelectOptionsHTML(config) {
    function makeOption(key) {
        return '<option value="' + key + '"' + ((key === smCount) ? 'selected' : '') + '>' + key + ' SM' + ((key > 1) ? 's' : '') + '</option>';
    }
    let optionsHTML = Array.from(new Array(config.max - config.min + 1), (_, i) => makeOption(i + config.min));
    return optionsHTML.join("\n");
}

function makeCacheSizeSelectOptionsHTML(config) {
    function makeOption(key) {
        return '<option value="' + key + '"' + ((key === cacheLineCount) ? 'selected' : '') + '>Cache ' + ((key > 0) ? 'enabled' : 'disabled') + '</option>';
    }
    let optionsHTML = [
        makeOption(0),
        makeOption(CONFIG.cache.L2CacheLines),
    ];
    return optionsHTML.join("\n");
}

function makeLatencySelectOptionsHTML() {
    function makeOption(key) {
        return '<option value="' + key + '"' + ((key === instructionLatencies) ? 'selected' : '') + '>' + CONFIG.latencies[key].name + ' latency</option>';
    }
    return Array.from(Object.keys(CONFIG.latencies), makeOption).join("\n");
}

function makeHighlightSelectOptionsHTML() {
    function makeOption(key) {
        return '<option value="' + key + '"' + ((key === highlightKernelLines) ? 'selected' : '') + '>Highlighted kernel lines ' + key + ' </option>';
    }
    return [makeOption("on"), makeOption("off")].join("\n");
}

function makeMemorySlotSizeSelectOptionsHTML(config) {
    function makeOption(key) {
        return '<option value="' + key + '"' + ((key === memorySlotSize) ? 'selected' : '') + '>DRAM slot size ' + key + '</option>';
    }
    const options = [];
    for (let size = config.min; size <= config.max; size += config.step) {
        options.push(makeOption(size));
    }
    return options.join("\n");
}

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

function resetSize(element, newWidth, newHeight) {
    element.width = newWidth;
    element.height = newHeight;
    element.style.width = element.width.toString() + "px";
    element.style.height = element.height.toString() + "px";
}

// Define menubar buttons for interacting with the animation
function initUI() {
    const pauseButton = document.getElementById("pause-button");
    const restartButton = document.getElementById("restart-button");
    const kernelSelect = document.getElementById("kernel-select");
    const smCountSelect = document.getElementById("sm-count-select");
    const cacheSizeSelect = document.getElementById("cache-size-select");
    const latencySelect = document.getElementById("latency-select");
    const highlightSelect = document.getElementById("highlight-select");
    const memorySlotSizeSelect = document.getElementById("memory-slot-size-select");

    pauseButton.addEventListener("click", _ => {
        pause();
        pauseButton.value = drawing ? "Pause" : "Continue";
    });
    restartButton.addEventListener("click", _ => {
        pauseButton.value = "Pause";
        restart();
    });
    kernelSelect.addEventListener("change", event => {
        drawing = false;
        activeKernel = event.target.value;
        pauseButton.value = "Pause";
        restart();
    });
    smCountSelect.addEventListener("change", event => {
        drawing = false;
        smCount = parseInt(event.target.value);
        pauseButton.value = "Pause";
        restart();
    });
    cacheSizeSelect.addEventListener("change", event => {
        drawing = false;
        cacheLineCount = parseInt(event.target.value);
        pauseButton.value = "Pause";
        restart();
    });
    latencySelect.addEventListener("change", event => {
        drawing = false;
        instructionLatencies = event.target.value;
        pauseButton.value = "Pause";
        restart();
    });
    highlightSelect.addEventListener("change", event => {
        highlightKernelLines = event.target.value;
        device.setKernelHighlighting(highlightKernelLines === "on");
        clear(kernelCanvas, "hard");
    });
    memorySlotSizeSelect.addEventListener("change", event => {
        drawing = false;
        memorySlotSize = parseInt(event.target.value);
        pauseButton.value = "Pause";
        restart();
    });
}

function populateUI() {
    // SM list contents
    document.getElementById("sm-list").innerHTML = makeSMlistBody(smCount);
    // CUDA kernel selector
    document.getElementById("kernel-select").innerHTML = makeKernelSelectOptionsHTML(CUDAKernels);
    // Streaming multiprocessor count selector
    document.getElementById("sm-count-select").innerHTML = makeSMCountSelectOptionsHTML(CONFIG.SM.count);
    // Cache size selector
    document.getElementById("cache-size-select").innerHTML = makeCacheSizeSelectOptionsHTML(CONFIG.cache.L2CacheLines);
    // Instruction latency selector
    document.getElementById("latency-select").innerHTML = makeLatencySelectOptionsHTML();
    // Kernel line highlighting selector
    document.getElementById("highlight-select").innerHTML = makeHighlightSelectOptionsHTML();
    // Memory slot size selector
    document.getElementById("memory-slot-size-select").innerHTML = makeMemorySlotSizeSelectOptionsHTML(CONFIG.memory.slotSizes);
}

function initSimulation() {
    populateUI();

    memoryCanvasInput = document.getElementById("memoryCanvasInput");
    //memoryCanvasOutput = document.getElementById("memoryCanvasOutput");
    kernelCanvas = document.getElementById("kernelCanvas");

    // Choose default kernel
    const kernel = CUDAKernels[activeKernel];
    // Render kernel source to set pre-element size
    const kernelSource = document.getElementById("kernelSource");
    kernelSource.innerText = kernel.sourceLines.join('\n');
    // Align kernel source highlighting over the pre-element containing the source
    resetSizeFromElement(kernelSource, kernelCanvas);

    const sourceStyle = window.getComputedStyle(kernelSource, null);
    const sourceLineHeight = parseStyle(sourceStyle, "line-height", "em");

    // Initialize simulated GPU
    device = new Device(memoryCanvasInput, smCount, cacheLineCount, kernel.memory.input, memorySlotSize, kernel.memory.extraRowPadding);
    const grid = new Grid(kernel.grid.dimGrid, kernel.grid.dimBlock);
    const kernelArgs = Object.assign(kernel.kernelArgs, {
        output: function() { },
        input: device.memoryTransaction.bind(device, "get"),
    });
    const program = {
        sourceLines: kernel.sourceLines,
        sourceLineHeight: sourceLineHeight,
        statements: kernel.statements,
        kernelArgs: kernelArgs,
    };
    if (kernel.sourceLines.length - 2 !== kernel.statements.length) {
        console.error("WARNING: Inconsistent kernel source line count when compared to callable statements, expected " + (kernel.statements.length) + " source lines but got " + (kernel.sourceLines.length - 2));
    }
    device.setProgram(grid, program);
    device.setKernelHighlighting(highlightKernelLines === "on");

    // Insert custom kernel descriptions/messages above canvases
    for (let msgID of ["SMMessages", "memoryMessages", "sourceMessages"]) {
        const element = document.getElementById(msgID);
        const messages = kernel[msgID];
        if (typeof messages !== "undefined") {
            element.innerHTML = messages.join("\n\n");
            element.hidden = false;
        } else {
            // Always hide possible, previous message
            element.innerHTML = '';
            element.hidden = true;
        }
    }

    // Resize memory canvas and its container depending on the input array size
    const memoryCanvasContainer = document.getElementById("memoryCanvasContainer");
    const canvasWidth = kernel.memory.input.columns * (CONFIG.memory.slotPadding + memorySlotSize);
    let canvasHeight = kernel.memory.input.rows * (CONFIG.memory.slotPadding + memorySlotSize);
    if (typeof kernel.memory.extraRowPadding !== "undefined") {
        canvasHeight += kernel.memory.extraRowPadding.amount - CONFIG.memory.slotPadding;
    }
    resetSize(memoryCanvasInput, canvasWidth, canvasHeight);
    resetSize(memoryCanvasContainer, canvasWidth, memoryCanvasContainer.height + canvasHeight);
}

function clear(canvas, type) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = 'rgba(255, 255, 255, ' + ((type === "hard") ? 1.0 : 0.7) + ')';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function cancelDraw() {
    window.cancelAnimationFrame(prevRenderFrameID);
}

function queueDraw() {
    prevRenderFrameID = window.requestAnimationFrame(draw);
}

function draw(now) {
    clear(memoryCanvasInput, "hard");
    //clear(memoryCanvasOutput);
    clear(kernelCanvas);
    device.step();
    if (device.programTerminated()) {
        device.clear();
        clear(memoryCanvasInput, "hard");
        device.step();
        clear(kernelCanvas, "hard");
        drawing = false;
    }
    if (drawing) {
        queueDraw();
    }
}

function restart() {
    drawing = false;
    cancelDraw();
    initSimulation();
    drawing = true;
    queueDraw();
}

function pause() {
    drawing = !drawing;
    if (drawing) {
        queueDraw();
    }
}

document.addEventListener("DOMContentLoaded", _ => {
    initUI();
    restart();
});
