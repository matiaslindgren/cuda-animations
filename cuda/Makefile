NVCC=nvcc
NVCCFLAGS=--std=c++14 -O3

SOURCES=main.cu

all:
	mkdir --parents bin && $(NVCC) $(NVCCFLAGS) $(SOURCES) -o bin/app

clean:
	rm -rf bin
