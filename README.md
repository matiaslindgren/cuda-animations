Animations of CUDA GPU memory access patterns.
Runs in web browsers that support the HTML5 [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API).

The main goal of this project is to provide intuitive visualizations of the challenges related to data locality when programming CUDA-supported GPUs.
As a consequence, many relevant hardware-level details have been omitted in favor of simplicity.

## Global memory access

One of the most important performance aspects to consider when writing CUDA programs for GPUs is the theoretical memory bandwidth, which greatly limits the performance of GPUs [1][2][3].
Mei and Chu point out that while NVIDIA's GTX 980 has a computational power of 4612 GFlop/s, its theoretical memory bandwidth is only 224 GB/s [4].
They argue that the actual memory bandwidth is even lower, which makes it such a significant bottleneck [1].
It seems that this gap has only increased with NVIDIA's GTX 1080 card, which NVIDIA reports having a computational power of 8873 GFlop/s, while the theoretical memory bandwidth is limited to 320 GB/s [5].

When a thread warp requests access to global memory, the amount of memory transactions generated depends on the alignment of the data being accessed [2].
In the best case scenario, all 32 threads of a warp access consecutive, naturally aligned 4-byte words.
All words fit neatly into one 128-byte cache line and the GPU can fetch all words using a single 128-byte memory transaction.
However, if the memory accesses are scattered, the GPU has to access the data using multiple transactions, which reduces the memory throughput.

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
