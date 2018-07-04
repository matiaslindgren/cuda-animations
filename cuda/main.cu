#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCKSIZE 32

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

float next_float() {
    static std::random_device rd;
    static std::default_random_engine e(rd());
    static std::uniform_real_distribution<float> floats(0.0, 1.0);
    return floats(e);
}

__global__ void kernel(const float *in, float *out, int n) {
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


int main() {
    const size_t n = BLOCKSIZE << 7;
    const size_t data_size = n * n * sizeof(float);

    // Fill vector with random floats from uniform(0, 1)
    std::vector<float> data(n*n);
    std::generate(data.begin(), data.end(), next_float);

    float* device_data = NULL;
    CHECK(cudaMalloc(&device_data, data_size));
    CHECK(cudaMemcpy(device_data, data.data(), data_size, cudaMemcpyHostToDevice));

    float* device_result = NULL;
    CHECK(cudaMalloc(&device_result, data_size));

    dim3 dimGrid(n/BLOCKSIZE, n/BLOCKSIZE);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);

    kernel<<<dimGrid, dimBlock>>>(device_data, device_result, n);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(data.data(), device_result, data_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(device_data));
    CHECK(cudaFree(device_result));

}
