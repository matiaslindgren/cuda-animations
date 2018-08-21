"use strict";

function failHard() {
    drawing = false;
    const errorBanner = document.getElementById("body-error-banner");
    errorBanner.innerHTML = "Something went wrong, please see the developer console";
    errorBanner.hidden = false;
}

function assert(expr, msg) {
    if (!expr) {
        throw "ASSERTION FAILED: " + msg;
    }
}

function printobj(o) {
    console.log(JSON.stringify(o));
}

function generatePaletteRGBA(shadeCount) {
    const baseRGBA = [20, 20, 20, 0.2];
    // Set to non-zero if warp schedulers within an SM should have different shades of the SM color
    // const shadeIncrement = 255 / shadeCount;
    const shadeIncrement = 0;
    return Array.from(new Array(2), (_, component) => {
        return Array.from(new Array(shadeCount), (_, shade) => {
            const color = baseRGBA.slice();
            color[component] = 255 - Math.floor(shade * shadeIncrement);
            return color;
        });
    });
}

const CONFIG = {
    animation: {
        // Delay in ms between rendering each frame
        drawDelayMS: 1000.0 / 400.0,
        SMCycleLabelSize: 20,
        kernelHighlightPalette: generatePaletteRGBA(2),
    },
    latencies: {
        arithmetic: 5,
        L2CacheAccess: 15,
        memoryAccess: 40,
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
        coolDownPeriod: 15,
    },
    cache: {
        // Size of a L2 cacheline in slots
        L2CacheLineSize: 8,
        L2CacheLines: 4*8,
        cachedStateRGBA: [60, 60, 240, 0.5],
        pendingStateRGBA: [60, 60, 240, 0.2],
    },
    SM: {
        count: 2,
        warpSize: 32,
        warpSchedulers: 2,
        // The amount of animation render frames simulating one multiprocessor cycle
        framesPerSMCycle: 1,
        paddingX: 20,
        paddingY: 20,
        height: 105,
        colorRGBA: [100, 100, 100, 0.8],
    },
};

const CUDAKernels = {
    minPath: {
        displayName: "Minimum path",
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
                    this.jump(-5);
                }
            },
            function() {
                this.identity(0);
                //this.arraySet(this.args.output, this.args.n * this.locals.i + this.locals.j, this.locals.v);
            },
        ],
    },
    trivial: {
        displayName: "Trivial linear",
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
            "__global__ void kernel(const float* input, int n) {",
            "    const int i = threadIdx.x + blockIdx.x * blockDim.x;",
            "    const int j = threadIdx.y + blockIdx.y * blockDim.y;",
            "    float x = input[n*i + j];",
            "    float y = x + x;",
            "}",
        ],
        statements: [
            function() {
                this.locals.i = this.arithmetic(this.threadIdx.x + this.blockIdx.x * this.blockDim.x);
            },
            function() {
                this.locals.j = this.arithmetic(this.threadIdx.y + this.blockIdx.y * this.blockDim.y);
            },
            function() {
                this.locals.x = this.arrayGet(this.args.input, this.args.n * this.locals.i + this.locals.j);
            },
            function() {
                this.locals.y = this.arithmetic(this.locals.x + this.locals.x);
            },
        ],
    },
};

