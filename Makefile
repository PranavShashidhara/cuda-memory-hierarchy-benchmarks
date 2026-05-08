# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — CUDA Benchmark Suite
#  Target : Jetson Orin Nano (Ampere sm_87)
# ─────────────────────────────────────────────────────────────────────────────

NVCC := nvcc
CXX  := g++

# ─────────────────────────────────────────────────────────────────────────────
#  Shared compiler flags
# ─────────────────────────────────────────────────────────────────────────────

COMMON_FLAGS := -arch=sm_87 \
                -O3 \
                --use_fast_math \
                --ptxas-options=-v \
                --expt-relaxed-constexpr \
                -std=c++17 \
                -lineinfo

# ─────────────────────────────────────────────────────────────────────────────
#  Original build
# ─────────────────────────────────────────────────────────────────────────────

ORIG_TARGET := build/cuda_bench

ORIG_FLAGS := $(COMMON_FLAGS) \
              -Iinclude

ORIG_SRCS := baseline/main.cu \
             baseline/memory_kernels.cu \
             baseline/matmul_kernels.cu \
             baseline/wmma_matmul_kernels.cu

ORIG_HEADERS := baseline/kernels.cuh \
                baseline/utils.cuh

ORIG_RESULTS := baseline

# ─────────────────────────────────────────────────────────────────────────────
#  Optimized build (Tensor Core version)
# ─────────────────────────────────────────────────────────────────────────────

OPT_TARGET := build/cuda_bench_optimized

OPT_FLAGS := $(COMMON_FLAGS) \
             -Iinclude

OPT_SRCS := Optimized/main.cu \
            Optimized/memory_kernels.cu \
            Optimized/matmul_kernels.cu \
            Optimized/wmma_matmul_kernels.cu 

OPT_HEADERS := Optimized/kernels.cuh \
               Optimized/utils.cuh

OPT_RESULTS := Optimized/results

# ─────────────────────────────────────────────────────────────────────────────
#  Targets
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all original optimized \
        run run-original run-optimized \
        plot bench compare \
        clean clean-original clean-optimized \
        nsys ncu dirs

all: dirs original optimized

# ─────────────────────────────────────────────────────────────────────────────
#  Directories
# ─────────────────────────────────────────────────────────────────────────────

dirs:
	@mkdir -p build
	@mkdir -p build/optimized
	@mkdir -p $(ORIG_RESULTS)
	@mkdir -p $(OPT_RESULTS)

# ─────────────────────────────────────────────────────────────────────────────
#  Build — Original
# ─────────────────────────────────────────────────────────────────────────────

original: $(ORIG_TARGET)

$(ORIG_TARGET): $(ORIG_SRCS) $(ORIG_HEADERS)
	$(NVCC) $(ORIG_FLAGS) $(ORIG_SRCS) -o $(ORIG_TARGET)
	@echo "Built: $(ORIG_TARGET)"

# ─────────────────────────────────────────────────────────────────────────────
#  Build — Optimized
# ─────────────────────────────────────────────────────────────────────────────

optimized: $(OPT_TARGET)

$(OPT_TARGET): $(OPT_SRCS) $(OPT_HEADERS)
	$(NVCC) $(OPT_FLAGS) $(OPT_SRCS) -o $(OPT_TARGET)
	@echo "Built: $(OPT_TARGET)"

# ─────────────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────────────

run: run-original run-optimized

run-original: original
	@echo "\n=== Running Original Benchmark ==="
	./$(ORIG_TARGET) | tee $(ORIG_RESULTS)/run.log

run-optimized: optimized
	@echo "\n=== Running Optimized Tensor Core Benchmark ==="
	./$(OPT_TARGET) | tee $(OPT_RESULTS)/run.log

# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

plot:
	python3 plot_results.py 

bench: run-optimized plot

compare: run plot

# ─────────────────────────────────────────────────────────────────────────────
#  Profiling
# ─────────────────────────────────────────────────────────────────────────────

nsys: optimized
	nsys profile \
	    --trace=cuda,nvtx \
	    --output=$(OPT_RESULTS)/report \
	    ./$(OPT_TARGET)

ncu: optimized
	ncu --set full \
	    --target-processes all \
	    --export $(OPT_RESULTS)/report_full \
	    ./$(OPT_TARGET)

# ─────────────────────────────────────────────────────────────────────────────
#  Clean
# ─────────────────────────────────────────────────────────────────────────────

clean: clean-original clean-optimized

clean-original:
	rm -f $(ORIG_TARGET)
	rm -f $(ORIG_RESULTS)/benchmark_results.csv
	rm -f $(ORIG_RESULTS)/run.log
	rm -f $(ORIG_RESULTS)/*.png

clean-optimized:
	rm -f $(OPT_TARGET)
	rm -f $(OPT_RESULTS)/benchmark_results.csv
	rm -f $(OPT_RESULTS)/run.log
	rm -f $(OPT_RESULTS)/*.png
	rm -f $(OPT_RESULTS)/report*