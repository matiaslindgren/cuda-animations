# CUDA GPU memory access patterns animated

Runs in web browsers that support the HTML5 [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API).

The main goal of this project is to provide intuitive visualizations of the challenges related to data locality when programming CUDA-supported GPUs.
As a consequence, many relevant hardware-level details have been omitted in the animations in favor of simplicity.

## Running locally

```
git clone --depth 1 https://github.com/matiaslindgren/cuda-animations
python3 -m http.server
```
Open a browser and go to localhost:8000.

You can try removing all assertion calls if the animation is running slow:
```
python3 build.py --build-dir build
cd build
python3 -m http.server
```


## GPU global memory access

One of the most significant aspects to consider when optimizing CUDA programs is the performance limitations caused by memory bandwidth during data transfer between the device memory and processing units [1][2][3].
When a thread warp requests access to global memory, the amount of memory transactions generated depends on the alignment of the data being accessed [2].
In the best case scenario, all 32 threads of a warp access consecutive, naturally aligned addresses of 4-byte words.
In this case, all words fit neatly into one 128-byte cache line and the GPU can fetch all words using a single 128-byte memory transaction.
However, if the memory accesses are scattered, the GPU has to access the data using multiple transactions, which reduces the memory throughput.

The image below, taken from [7], provides a clear illustration:

![Coalescing memory accesses](img/coalescing_mem_access.png "Coalescing memory accesses [7]")


### Computational power vs. memory bandwidth

Mei and Chu point out that while NVIDIA's GTX 980 (Maxwell architecture) has a computational power of 4612 GFlop/s, its theoretical memory bandwidth is only 224 GB/s [1][4].
It seems that this gap has only increased on NVIDIA's GTX 1080 card (Pascal architecture), which NVIDIA reports having a computational power of 8873 GFlop/s, while the theoretical memory bandwidth is limited to 320 GB/s [5].
On Volta cards, the computational power has been increased to 15700 GFlop/s for 32-bit floating point numbers, while the memory subsystem is reported to enable 900 GB/s peak memory bandwidth [6].

## Benchmarks

Some simple [benchmarking tools](cuda) are provided for running actual CUDA kernels of the simulated kernels.

Most recent results:
```
Quadro K2200

compute capability:           5
global memory bus width:      128 bits
streaming multiprocessors:    5
maximum threads per SM:       2048
L2 cache size:                2097152 bytes

    function   iteration  input size  time (μs)
-----------------------------------------------
     step_v0           1    16777216     7280259
     step_v0           2    16777216     7184349
     step_v0           3    16777216     7187062
     step_v0           4    16777216     7187541
     step_v0           5    16777216     7184044
     step_v1           1    16777216     1856605
     step_v1           2    16777216     1855964
     step_v1           3    16777216     1856726
     step_v1           4    16777216     1856108
     step_v1           5    16777216     1855794
     step_v2           1    16777216      374867
     step_v2           2    16777216      375245
     step_v2           3    16777216      374816
     step_v2           4    16777216      374948
     step_v2           5    16777216      374669
```

## References

[1] Xinxin Mei, and Xiaowen Chu.
"Dissecting GPU memory hierarchy through microbenchmarking."
IEEE Transactions on Parallel and Distributed Systems 28.1 (2017). pp. 72-86.
Available [online](https://arxiv.org/abs/1509.02308).

[2] "CUDA C Programming Guide."
NVIDIA Corporation, version PG-02829-001_v9.2 (May 2018).
Available [online](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

[3] "CUDA C Best Practices Guide."
NVIDIA Corporation, version DG-05603-001_v9.2 (May 2018).
Available [online](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

[4] "GeForce GTX 980 Whitepaper"
NVIDIA Corporation, (2014).
Available [online](https://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_980_Whitepaper_FINAL.PDF)

[5] "GeForce GTX 1080 Whitepaper"
NVIDIA Corporation, (2016).
Available [online](https://international.download.nvidia.com/geforce-com/international/pdfs/GeForce_GTX_1080_Whitepaper_FINAL.pdf)

[6] "Tesla V100 Architecture"
NVIDIA Corporation, (August 2017).
Available [online](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

[7] Volkov Vasily
"Understanding Latency Hiding on GPUs"
UC Berkeley, (2016).
ProQuest ID: Volkov_berkeley_0028E_16465. Merritt ID: ark:/13030/m5f52bdd.
Available [online](https://escholarship.org/uc/item/1wb7f3h4)
