"use strict";

function failHard() {
    drawing = false;
    const errorBanner = document.getElementById("body-error-banner");
    errorBanner.innerHTML = "Something went wrong, please see the developer console";
    errorBanner.hidden = false;
    cancelDraw();
}

function assert(expr, msg, state) {
    if (!expr) {
        failHard();
        console.error("ASSERTION FAILED");
        if (typeof state !== "undefined") {
            console.error(state.name + " was:");
            printobj(state.obj);
        }
        throw "AssertionError: " + msg;
    }
}

function printobj(o) {
    console.log(JSON.stringify(o, null, 2));
}


function get4Palette(key) {
    switch(key) {
        case "rgba-colorful":
            const alpha = 0.25;
            return [
                [35, 196, 1, alpha],
                [47, 21, 162, alpha],
                [227, 1, 23, alpha],
                [235, 190, 2, alpha],
            ];
        default:
            failHard();
            console.error("unknown palette key: " + key);
    }
}


const CONFIG = {
    animation: {
        // Delay in ms between rendering each frame
        drawDelayMS: 1000.0 / 400.0,
        // Array of colors for highlighting kernel source lines and SMs
        kernelHighlightPalette: get4Palette("rgba-colorful"),
    },
    // Simulated latency in cycles for different instruction types
    latencies: {
        arithmetic: 2,
        L2CacheAccess: 4,
        memoryAccess: 16,
    },
    memory: {
        // Amount of indexable memory slots on each row and column
        rowSlotCount: 32,
        columnSlotCount: 32,
        // Empty space between each slot on all sides
        slotPadding: 1,
        slotSize: 14,
        slotFillRGBA: [100, 100, 100, 0.15],
        // Amount of animation steps of the cooldown transition after touching a memory index
        coolDownPeriod: 10,
    },
    cache: {
        // Size of a L2 cacheline in slots
        L2CacheLineSize: 8,
        L2CacheLines: 8*8,
        cachedStateRGBA: [60, 60, 200, 0.5],
        pendingStateRGBA: [60, 60, 200, 0.2],
    },
    SM: {
        count: {
            default: 2,
            min: 1,
            max: 4,
        },
        warpSize: 32,
        warpSchedulers: 2,
        // Probability of skipping an SM cycle (for simulating hardware latency)
        // Setting this to 0 makes all SMs to execute in unison
        latencyNoiseProb: 0.05,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 1,
    },
};

const CUDAKernels = {
    ppcStep: {
        displayName: "Shortcut step",
        kernelArgsN: 32,
        grid: {
            dimGrid: {
                x: Math.round(32/8),
                y: Math.round(32/8),
            },
            dimBlock: {
                x: 8,
                y: 8
            },
        },
        sourceLines: [
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
        ],
        // Closures that simulate the CUDA statements above
        // Each closure is applied with a CUDA context, which can then be referenced as 'this' in the closure
        statements: [
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
                if (this.locals.k < this.args.n) {
                    // Jump instructions take an integer line offset as parameter
                    // In this case, -4 jumps to the function that assigns to this.locals.x
                    this.jump(-4);
                }
            },
            function() {
                this.identity(0);
                //this.arraySet(this.args.output, this.args.n * this.locals.i + this.locals.j, this.locals.v);
            },
        ],
    },
    trivial: {
        displayName: "Fully coalesced",
        grid: {
            dimGrid: {
                x: 1,
                y: 32,
            },
            dimBlock: {
                x: 32,
                y: 1,
            },
        },
        sourceLines: [
            "__global__ void kernel(float* output, const float* input) {",
            "    const float c = 2.0;",
            "    const int i = threadIdx.x + blockIdx.y * blockDim.x;",
            "    float x = input[i];",
            "    output[i] = c * x;",
            "}",
        ],
        statements: [
            function() {
                this.locals.c = this.identity(2.0);
            },
            function() {
                this.locals.i = this.arithmetic(this.threadIdx.x + this.blockIdx.y * this.blockDim.x);
            },
            function() {
                this.locals.x = this.arrayGet(this.args.input, this.locals.i);
            },
            function() {
                this.arithmetic(this.locals.c * this.locals.x);
            },
        ],
    },
};

