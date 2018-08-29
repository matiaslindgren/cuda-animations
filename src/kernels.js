const ppcStepV0Lines = [
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

// Closures that simulate the CUDA statements above
// Each closure is applied with a CUDA context, which can then be referenced as 'this' in the closure
const ppcStepV0Statements = [
function() { this.locals.i = this.arithmetic(this.threadIdx.x + this.blockIdx.x * this.blockDim.x); },
function() { this.locals.j = this.arithmetic(this.threadIdx.y + this.blockIdx.y * this.blockDim.y); },
function() { this.locals.v = this.identity(Infinity); },
function() { this.locals.k = this.identity(0); },
function() { this.locals.x = this.arrayGet(this.args.input, this.args.n * this.locals.i + this.locals.k); },
function() { this.locals.y = this.arrayGet(this.args.input, this.args.n * this.locals.k + this.locals.j); },
function() { this.locals.z = this.arithmetic(this.locals.x + this.locals.y); },
function() { this.locals.v = this.arithmetic(Math.min(this.locals.v, this.locals.z)); },
function() {
    // Jump instructions take an integer line offset as parameter
    // In this case, -4 jumps to the function that assigns to this.locals.x
    if (++this.locals.k < this.args.n) {
        this.jump(-4);
    }
},
function() { this.identity(0); },
];

const ppcStepV1Lines = [
"__global__ void kernel(float* output, const float* input, int n) {",
"    const int i = threadIdx.x + blockIdx.x * blockDim.x;",
"    const int j = threadIdx.y + blockIdx.y * blockDim.y;",
"    float v = HUGE_VALF;",
"    for (int k = 0; k < n; ++k) {",
"        float x = input[n*j + k];",
"        float y = input[n*k + i];",
"        float z = x + y;",
"        v = min(v, z);",
"    }",
"    output[n*j + i] = v;",
"}",
];

const ppcStepV1Statements = [
function() { this.locals.i = this.arithmetic(this.threadIdx.x + this.blockIdx.x * this.blockDim.x); },
function() { this.locals.j = this.arithmetic(this.threadIdx.y + this.blockIdx.y * this.blockDim.y); },
function() { this.locals.v = this.identity(Infinity); },
function() { this.locals.k = this.identity(0); },
function() { this.locals.x = this.arrayGet(this.args.input, this.args.n * this.locals.j + this.locals.k); },
function() { this.locals.y = this.arrayGet(this.args.input, this.args.n * this.locals.k + this.locals.i); },
function() { this.locals.z = this.arithmetic(this.locals.x + this.locals.y); },
function() { this.locals.v = this.arithmetic(Math.min(this.locals.v, this.locals.z)); },
function() { if (++this.locals.k < this.args.n) { this.jump(-4); } },
function() { this.identity(0); },
];

const ppcStepV2Lines = [
"__global__ void kernel(float* output, const float* input, int n) {",
"    const int ia = threadIdx.x;",
"    const int ja = threadIdx.y;",
"    const int ic = blockIdx.x;",
"    const int jc = blockIdx.y;",
" ",
"    const float* t = d + n * n;",
" ",
"    float v[8]v[8];",
"    for (int ib = 0; ib < 8; ++ib) {",
"        for (int jb = 0; jb < 8; ++jb) {",
"            v[ib][jb] = HUGE_VALF;",
"        }",
"    }",
"    for (int k = 0; k < n; ++k) {",
"        float x[8];",
"        float y[8];",
"        for (int ib = 0; ib < 8; ++ib) {",
"            int i = ic * 64 + ib * 8 + ia;",
"            x[ib] = t[n*k + i];",
"        }",
"        for (int jb = 0; jb < 8; ++jb) {",
"            int j = jc * 64 + jb * 8 + ja;",
"            y[jb] = d[n*k + j];",
"        }",
"        for (int ib = 0; ib < 8; ++ib) {",
"            for (int jb = 0; jb < 8; ++jb) {",
"                v[ib][jb] = min(v[ib][jb], x[ib] + y[jb]);",
"            }",
"        }",
"    }",
"    for (int ib = 0; ib < 8; ++ib) {",
"        for (int jb = 0; jb < 8; ++jb) {",
"            int i = ic * 64 + ib * 8 + ia;",
"            int j = jc * 64 + jb * 8 + ja;",
"            output[n*i + j] = v[ib][jb];",
"        }",
"    }",
"}",
];

const ppcStepV2Statements = [
function() { this.locals.ia = this.identity(this.threadIdx.x); },
function() { this.locals.ja = this.identity(this.threadIdx.y); },
function() { this.locals.ic = this.identity(this.blockIdx.x); },
function() { this.locals.jc = this.identity(this.blockIdx.y); },
function() { this.identity(null); },
function() { this.locals.t = this.identity("arrayHandle"); },
function() { this.identity(null); },
function() { this.locals.v = this.identity(Array.from(new Array(8), _ => new Array(8))); },
function() { this.locals.ib = this.identity(0); },
function() { this.locals.jb = this.identity(0); },
function() {
    const ib = this.locals.ib;
    const jb = this.locals.jb;
    const v = this.locals.v;
    v[ib][jb] = this.identity(Infinity);
    assert(v.every(vv => vv.length === 8));
},
function() { if (++this.locals.jb < 8) { this.jump(-1); } else { this.locals.jb = undefined; } },
function() { if (++this.locals.ib < 8) { this.jump(-3); } else { this.locals.ib = undefined; } },
function() { this.locals.k = this.identity(0); },
function() { this.locals.x = this.identity(new Array(8)); },
function() { this.locals.y = this.identity(new Array(8)); },
function() { this.locals.ib = this.identity(0); },
function() {
    const ia = this.locals.ia;
    const ib = this.locals.ib;
    const ic = this.locals.ic;
    this.locals.i = this.arithmetic(ic * 64 + ib * 8 + ia);
},
function() {
    const ib = this.locals.ib;
    assert(ib < 8, "ib too large");
    const n = this.args.n;
    const k = this.locals.k;
    assert(k < 32, "k too large");
    const i = this.locals.i;
    const x = this.locals.x;
    x[ib] = this.arrayGet(this.args.input, n*n + n*k + i); // Get from t
},
function() { if (++this.locals.ib < 8) { this.jump(-2); } else { this.locals.ib = undefined; } },
function() { this.locals.jb = this.identity(0); },
function() {
    assert(this.locals.jb < 8, "jb too large");
    const ja = this.locals.ja;
    const jb = this.locals.jb;
    const jc = this.locals.jc;
    this.locals.j = this.arithmetic(jc * 64 + jb * 8 + ja);
},
function() {
    const jb = this.locals.jb;
    const n = this.args.n;
    const k = this.locals.k;
    assert(k < 32, "k too large");
    const j = this.locals.j;
    const y = this.locals.y;
    y[jb] = this.arrayGet(this.args.input, n*k + j); // Get from d
},
function() { if (++this.locals.jb < 8) { this.jump(-2); } else { this.locals.jb = undefined; } },
function() { this.locals.ib = this.identity(0); },
function() { this.locals.jb = this.identity(0); },
function() {
    const ib = this.locals.ib;
    const jb = this.locals.jb;
    const v = this.locals.v;
    const x = this.locals.x;
    const y = this.locals.y;
    v[ib][jb] = this.arithmetic(Math.min(v[ib][jb], x[ib] + y[jb]));
},
function() { if (++this.locals.jb < 8) { this.jump(-1); } else { this.locals.jb = undefined; } },
function() { if (++this.locals.ib < 8) { this.jump(-3); } else { this.locals.ib = undefined; } },
function() { if (++this.locals.k < this.args.n) { this.jump(-15); } else { this.locals.k = undefined; } },
function() { this.locals.ib = this.identity(0); },
function() { this.locals.jb = this.identity(0); },
function() {
    const ia = this.locals.ia;
    const ib = this.locals.ib;
    const ic = this.locals.ic;
    this.locals.i = this.arithmetic(ic * 64 + ib * 8 + ia);
},
function() {
    const ja = this.locals.ja;
    const jb = this.locals.jb;
    const jc = this.locals.jc;
    this.locals.j = this.arithmetic(jc * 64 + jb * 8 + ja);
},
function() {
    const n = this.args.n;
    const i = this.locals.i;
    const j = this.locals.j;
    this.arithmetic(n*i + j); // FIXME assign to output array
},
function() { if (++this.locals.jb < 8) { this.jump(-3); } else { this.locals.jb = undefined; } },
function() { if (++this.locals.ib < 8) { this.jump(-5); } else { this.locals.ib = undefined; } },
];

const CUDAKernels = {

    ppcStepV0: {
        displayName: "Shortcut step v0",
        kernelArgs: {
            n: 32,
        },
        memory: {
            input: {
                rows: 32,
                columns: 32,
            },
        },
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
        sourceLines: ppcStepV0Lines,
        statements: ppcStepV0Statements,
    },

    ppcStepV1: {
        displayName: "Shortcut step v1",
        kernelArgs: {
            n: 32,
        },
        memory: {
            input: {
                rows: 32,
                columns: 32,
            },
        },
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
        sourceLines: ppcStepV1Lines,
        statements: ppcStepV1Statements,
    },

    ppcStepV2: {
        displayName: "Shortcut step v2",
        kernelArgs: {
            n: 32,
        },
        memory: {
            input: {
                rows: 64,
                columns: 32,
            },
        },
        grid: {
            dimGrid: {
                x: 2,
                y: 2,
            },
            dimBlock: {
                x: 8,
                y: 8
            },
        },
        sourceLines: ppcStepV2Lines,
        statements: ppcStepV2Statements,
    },

    trivial: {
        displayName: "Fully coalesced",
        kernelArgs: {},
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
        memory: {
            input: {
                rows: 32,
                columns: 32,
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
