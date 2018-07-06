NVCC=nvcc
NVCCFLAGS=--std=c++14 -O3

MAIN_SRC=main.cu
ANALYZE_SRC=analyze.cu

all: dirs app analyze

dirs:
	mkdir --parents bin

app:
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) -o bin/app

analyze:
	$(NVCC) $(NVCCFLAGS) $(ANALYZE_SRC) -o bin/analyze

clean:
	rm -rf bin
