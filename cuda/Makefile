NVCC=nvcc
NVCCFLAGS=--std=c++14 -O3

BENCHMARK_SRC=benchmark.cu
ANALYZE_SRC=analyze.cu

all: dirs benchmark analyze

dirs:
	mkdir --parents bin

benchmark:
	$(NVCC) $(NVCCFLAGS) $(BENCHMARK_SRC) -o bin/benchmark

analyze:
	$(NVCC) $(NVCCFLAGS) $(ANALYZE_SRC) -o bin/analyze

clean:
	rm -rf bin
