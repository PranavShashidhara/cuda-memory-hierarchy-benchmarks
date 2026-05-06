NVCC        = nvcc
NCU         = /usr/local/cuda-12.6/bin/ncu
NVCCFLAGS   = -std=c++17 -arch=sm_87 -diag-suppress 177
OPT_FLAGS   = -O3 -use_fast_math
DEBUG_FLAGS = -G -g

SRC_DIR     = src
KERNEL_DIR  = kernels
INC_DIR     = include
OPT_DIR     = Optimized
BUILD_DIR   = build
OPT_BUILD   = build/optimized

TARGET      = cuda_bench
OPT_TARGET  = cuda_bench_opt

SRCS        = $(SRC_DIR)/main.cu $(SRC_DIR)/utils.cu \
              $(KERNEL_DIR)/matmul_kernels.cu $(KERNEL_DIR)/memory_kernels.cu

OPT_SRCS    = $(OPT_DIR)/main.cu \
              $(OPT_DIR)/matmul_kernels.cu $(OPT_DIR)/memory_kernels.cu

.PHONY: all build optimized run run-opt profile profile-nsys clean

all: build optimized

build: $(SRCS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) $(SRCS) -o $(BUILD_DIR)/$(TARGET)
	@echo "Standard build done: $(BUILD_DIR)/$(TARGET)"

optimized: $(OPT_SRCS)
	mkdir -p $(OPT_BUILD)
	$(NVCC) $(NVCCFLAGS) $(OPT_FLAGS) -I$(OPT_DIR) $(OPT_SRCS) -o $(OPT_BUILD)/$(OPT_TARGET)
	@echo "Optimized build done: $(OPT_BUILD)/$(OPT_TARGET)"

run: build
	./$(BUILD_DIR)/$(TARGET)

run-opt: optimized
	./$(OPT_BUILD)/$(OPT_TARGET)

profile: optimized
	sudo env PATH=/usr/local/cuda-12.6/bin:$$PATH $(NCU) --set full -f -o report_clean ./$(OPT_BUILD)/$(OPT_TARGET)

profile-nsys: optimized
	sudo env PATH=/usr/local/cuda-12.6/bin:$$PATH nsys profile -f -o report ./$(OPT_BUILD)/$(OPT_TARGET)

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned."