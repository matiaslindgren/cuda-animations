// kernels from http://ppc.cs.aalto.fi/ch4 (2018)
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include <cstdio>
#include <cuda_runtime.h>

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

#define BLOCKSIZE 32


float next_float() {
    static std::random_device rd;
    static std::default_random_engine e(rd());
    static std::uniform_real_distribution<float> floats(0.0, 1.0);
    return floats(e);
}


inline int static divup(int a, int b) {
    return (a + b - 1)/b;
}


inline int static roundup(int a, int b) {
    return divup(a, b) * b;
}


__global__ void kernel_v0(const float *in, float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float v = HUGE_VALF;
    for (int k = 0; k < n; ++k) {
        float x = in[n*i + k];
        float y = in[n*k + j];
        float z = x + y;
        v = min(v, z);
    }
    out[n*i + j] = v;
}


__global__ void kernel_v1(const float *in, float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    float v = HUGE_VALF;
    for (int k = 0; k < n; ++k) {
        float x = in[n*j + k];
        float y = in[n*k + i];
        float z = x + y;
        v = min(v, z);
    }
    out[n*j + i] = v;
}


__global__ void kernel_v2(float* r, const float* d, int n, int nn) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    const float* t = d + nn * nn;

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = HUGE_VALF;
        }
    }
    for (int k = 0; k < n; ++k) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = t[nn*k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = d[nn*k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] = min(v[ib][jb], x[ib] + y[jb]);
            }
        }
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < n && j < n) {
                r[n*i + j] = v[ib][jb];
            }
        }
    }
}


__global__ void add_padding_v2(const float* r, float* d, int n, int nn) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    float* t = d + nn * nn;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < n && j < n) ? r[n*i + j] : HUGE_VALF;
        d[nn*i + j] = v;
        t[nn*j + i] = v;
    }
}


void step_v0(float* r, const float* d, int n) {
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    float* rGPU = NULL;

    CHECK(cudaMalloc(&dGPU, n * n * sizeof(float)));
    CHECK(cudaMalloc(&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    kernel_v0<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}


void step_v1(float* r, const float* d, int n) {
    float* dGPU = NULL;
    float* rGPU = NULL;

    CHECK(cudaMalloc(&dGPU, n * n * sizeof(float)));
    CHECK(cudaMalloc(&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    kernel_v1<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}


void step_v2(float* r, const float* d, int n) {
    int nn = roundup(n, 64);

    float* dGPU = NULL;
    float* rGPU = NULL;

    CHECK(cudaMalloc(&dGPU, 2 * nn * nn * sizeof(float)));
    CHECK(cudaMalloc(&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(rGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        add_padding_v2<<<dimGrid, dimBlock>>>(rGPU, dGPU, n, nn);
        CHECK(cudaGetLastError());
    }

    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn / 64, nn / 64);
        kernel_v2<<<dimGrid, dimBlock>>>(rGPU, dGPU, n, nn);
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}


void print_header() {
    std::cout << std::setw(12) << "function"
              << std::setw(12) << "iteration"
              << std::setw(12) << "input size"
              << std::setw(12) << "time (μs)"
              << std::endl;
}
void print_row(const char* name, int i, size_t n, double time) {
    std::cout << std::setw(12) << name
              << std::setw(12) << i
              << std::setw(12) << n*n
              << std::setw(12) << (int)(1e6 * time)
              << std::endl;
}


struct FunctionData {
    const char* name;
    void (*callable)(float*, const float*, int);
    const size_t n;
};


int main(int argc, char** argv) {
    int iterations = 1;
    if (argc > 1) {
        iterations = std::stoi(argv[1]);
    }

    std::vector<FunctionData> functions = {
        {"step_v0", step_v0, BLOCKSIZE << 7},
        {"step_v1", step_v1, BLOCKSIZE << 7},
        {"step_v2", step_v2, BLOCKSIZE << 7},
    };

    print_header();

    for (auto func : functions) {
        const size_t n = func.n;
        for (auto i = 0; i < iterations; ++i) {
            std::vector<float> data(n*n);
            std::generate(data.begin(), data.end(), next_float);
            std::vector<float> result(n*n);
            const auto time_start = std::chrono::high_resolution_clock::now();
            func.callable(result.data(), data.data(), n);
            const auto time_end = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<float> delta_seconds = time_end - time_start;
            print_row(func.name, i + 1, n, delta_seconds.count());
        }
    }

}
