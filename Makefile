NVCC = nvcc
CFLAGS = -Iinclude -O3

TARGET = cuda_bench

SRCS = src/main.cu src/utils.cu $(wildcard kernels/*.cu)

all:
	$(NVCC) $(SRCS) $(CFLAGS) -o $(TARGET)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET)