"use strict";

// All mutable global variables
var memoryCanvasInput;
//var memoryCanvasOutput;
var kernelCanvas;
var device;
var prevRenderTime = performance.now();
var drawing = true;

// Choose default CUDA kernel from kernels defined in config.js
var activeKernel = "ppcStep";
// Amount of streaming multiprocessors in device
var smCount = CONFIG.SM.count.default;

function makeSMlistBody(count) {
    function liWrap(s, liID) {
        if (typeof liID === "undefined") {
            return "<li>" + s + "</li>";
        } else {
            return "<li id=\"sm-state-" + liID + "\">" + s + "</li>";
        }
    }

    function SMcontentsToUL(body) {
        return "<ul>" + body.map(liWrap).join("\n") + "</ul>";
    }

    const defaultSMstateBody = [
        "<pre>cycle <span class=\"sm-cycle-counter\">0</span></pre>",
    ];
    const liElements = Array.from(new Array(count), (_, i) => {
        return liWrap(SMcontentsToUL(defaultSMstateBody), i + 1);
    });
    return liElements.join("\n");
}

function makeKernelSelectOptionsHTML(kernels) {
    function makeOption(key) {
        return "<option value=\"" + key + "\">" + kernels[key].displayName + "</option>";
    }
    // Create all kernels as options HTML, where the default kernel is first
    const kernelsNoDefault = Object.keys(kernels).filter(k => k !== activeKernel);
    const optionsHTML = Array.from([activeKernel].concat(kernelsNoDefault), makeOption);
    return optionsHTML.join("\n");
}

function makeSMCountSelectOptionsHTML(config) {
    function makeOption(key) {
        return '<option value="' + key + '"' + ((key === smCount) ? 'selected' : '') + '>' + key + ' SMs</option>';
    }
    let optionsHTML = Array.from(new Array(config.max - config.min + 1), (_, i) => makeOption(i + config.min));
    return optionsHTML.join("\n");
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

function init() {
    // Populate UI elements
    // SM list contents
    document.getElementById("sm-list").innerHTML = makeSMlistBody(smCount);
    // CUDA kernel selector
    document.getElementById("kernel-select").innerHTML = makeKernelSelectOptionsHTML(CUDAKernels);
    // Streaming multiprocessor count selector
    document.getElementById("sm-count-select").innerHTML = makeSMCountSelectOptionsHTML(CONFIG.SM.count);

    memoryCanvasInput = document.getElementById("memoryCanvasInput");
    //memoryCanvasOutput = document.getElementById("memoryCanvasOutput");
    kernelCanvas = document.getElementById("kernelCanvas");

    // Initialize canvas element dimensions from computed stylesheet
    [memoryCanvasInput].forEach(canvas => resetSizeAttrsFromStyle(canvas));

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
    device = new Device(memoryCanvasInput, smCount);
    const grid = new Grid(kernel.grid.dimGrid, kernel.grid.dimBlock);
    const kernelArgs = {
        output: function() { },
        input: device.memoryTransaction.bind(device, "get"),
        n: kernel.kernelArgsN || 0,
    };
    const program = {
        sourceLines: kernel.sourceLines,
        sourceLineHeight: sourceLineHeight,
        statements: kernel.statements,
        kernelArgs: kernelArgs,
    };
    if (kernel.sourceLines.length - 2 !== kernel.statements.length) {
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
    //clear(memoryCanvasOutput);
    clear(kernelCanvas);
    device.step();
    if (device.programTerminated()) {
        clear(memoryCanvasInput);
        device.clear();
        device.step();
        clear(kernelCanvas);
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
    const pauseButton = document.getElementById("pause-button");
    const restartButton = document.getElementById("restart-button");
    const kernelSelect = document.getElementById("kernel-select");
    const smCountSelect = document.getElementById("sm-count-select");
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
        restart();
    });
    smCountSelect.addEventListener("change", event => {
        drawing = false;
        smCount = parseInt(event.target.value);
        restart();
    });
}

document.addEventListener("DOMContentLoaded", _ => {
    initUI();
    restart();
});
