# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — CUDA Benchmark Suite
#  Target : Jetson Orin Nano (Ampere sm_87)
#
#  Builds:
#    original   — baseline fp32 + WMMA kernels
#    optimized  — optimized fp32 + WMMA kernels
#    cublas     — cuBLAS fp32 + fp16 TC hardware ceiling
#    cutlass    — CUTLASS Simt fp32 + TensorOp fp16 TC structured baseline
#    all        — all four targets
# ─────────────────────────────────────────────────────────────────────────────

NVCC := nvcc
CXX  := g++

# ─────────────────────────────────────────────────────────────────────────────
#  CUTLASS path — override on the command line if needed:
#    make cutlass CUTLASS_DIR=/path/to/cutlass
# ─────────────────────────────────────────────────────────────────────────────
CUTLASS_DIR ?= $(HOME)/cutlass

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
ORIG_TARGET  := build/cuda_bench
ORIG_FLAGS   := $(COMMON_FLAGS) -Iinclude
ORIG_SRCS    := baseline/main.cu \
                baseline/memory_kernels.cu \
                baseline/matmul_kernels.cu \
                baseline/wmma_matmul_kernels.cu
ORIG_HEADERS := baseline/kernels.cuh baseline/utils.cuh
ORIG_RESULTS := baseline/results

# ─────────────────────────────────────────────────────────────────────────────
#  Optimized build (Tensor Core version)
# ─────────────────────────────────────────────────────────────────────────────
OPT_TARGET  := build/cuda_bench_optimized
OPT_FLAGS   := $(COMMON_FLAGS) -Iinclude
OPT_SRCS    := Optimized/main.cu \
               Optimized/memory_kernels.cu \
               Optimized/matmul_kernels.cu \
               Optimized/wmma_matmul_kernels.cu
OPT_HEADERS := Optimized/kernels.cuh Optimized/utils.cuh
OPT_RESULTS := Optimized/results

# ─────────────────────────────────────────────────────────────────────────────
#  cuBLAS build
#  — links -lcublas, no extra include path needed (ships with CUDA toolkit)
#  — standalone binary; run independently or call run_cublas_baseline() from
#    main.cu after integrating the source into the optimized build
# ─────────────────────────────────────────────────────────────────────────────
CUBLAS_TARGET  := build/cuda_bench_cublas
CUBLAS_FLAGS   := $(COMMON_FLAGS) \
                  -IOptimized \
                  -DCUBLAS_STANDALONE
CUBLAS_SRCS    := cublass_and_cutlass/cublas_baseline.cu
CUBLAS_LIBS    := -lcublas
CUBLAS_RESULTS := cublass_and_cutlass/results

# ─────────────────────────────────────────────────────────────────────────────
#  CUTLASS build
#  — header-only on the CUDA side; no link library needed
#  — requires CUTLASS headers at $(CUTLASS_DIR)/include
#  — standalone binary; run independently or call run_cutlass_baseline() from
#    main.cu after integrating the source into the optimized build
# ─────────────────────────────────────────────────────────────────────────────
CUTLASS_TARGET  := build/cuda_bench_cutlass
CUTLASS_FLAGS   := $(COMMON_FLAGS) \
                   -IOptimized \
                   -I$(CUTLASS_DIR)/include \
                   -I$(CUTLASS_DIR)/tools/util/include \
                   -DCUTLASS_STANDALONE
CUTLASS_SRCS    := cublass_and_cutlass/cutlass_baseline.cu
CUTLASS_RESULTS := cublass_and_cutlass/results

# ─────────────────────────────────────────────────────────────────────────────
#  Phony targets
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all original optimized cublas cutlass clone-cutlass \
        run run-original run-optimized run-cublas run-cutlass \
        plot bench compare \
        clean clean-original clean-optimized clean-cublas clean-cutlass \
        nsys ncu dirs check-cutlass

all: dirs original optimized cublas cutlass

# ─────────────────────────────────────────────────────────────────────────────
#  Directories
# ─────────────────────────────────────────────────────────────────────────────
dirs:
	@mkdir -p build
	@mkdir -p build/optimized
	@mkdir -p $(ORIG_RESULTS)
	@mkdir -p $(OPT_RESULTS)
	@mkdir -p $(CUBLAS_RESULTS)
	@mkdir -p $(CUTLASS_RESULTS)

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
#  Build — cuBLAS baseline
# ─────────────────────────────────────────────────────────────────────────────
cublas: dirs $(CUBLAS_TARGET)

$(CUBLAS_TARGET): $(CUBLAS_SRCS) Optimized/utils.cuh
	$(NVCC) $(CUBLAS_FLAGS) $(CUBLAS_SRCS) $(CUBLAS_LIBS) -o $(CUBLAS_TARGET)
	@echo "Built: $(CUBLAS_TARGET)  (standalone, links -lcublas)"

# ─────────────────────────────────────────────────────────────────────────────
#  Build — CUTLASS baseline
#  Guard with a CUTLASS header check so the error message is helpful.
# ─────────────────────────────────────────────────────────────────────────────
check-cutlass:
	@test -f $(CUTLASS_DIR)/include/cutlass/cutlass.h || \
	  (echo "" && \
	   echo "ERROR: CUTLASS headers not found at $(CUTLASS_DIR)/include." && \
	   echo "       Run one of:" && \
	   echo "         make clone-cutlass                          # clones to ~/cutlass" && \
	   echo "         make cutlass CUTLASS_DIR=/path/to/cutlass  # use existing clone" && \
	   echo "" && exit 1)

# Clone CUTLASS (shallow, ~40 MB) into CUTLASS_DIR then immediately build.
clone-cutlass:
	@echo "Cloning CUTLASS into $(CUTLASS_DIR) ..."
	git clone --depth=1 https://github.com/NVIDIA/cutlass.git $(CUTLASS_DIR)
	@echo "CUTLASS cloned. Building cutlass target..."
	$(MAKE) cutlass CUTLASS_DIR=$(CUTLASS_DIR)

cutlass: dirs check-cutlass $(CUTLASS_TARGET)

$(CUTLASS_TARGET): $(CUTLASS_SRCS) Optimized/utils.cuh
	$(NVCC) $(CUTLASS_FLAGS) $(CUTLASS_SRCS) -o $(CUTLASS_TARGET)
	@echo "Built: $(CUTLASS_TARGET)  (standalone, CUTLASS header-only)"

# ─────────────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────────────
run: run-original run-optimized run-cublas run-cutlass

run-original: original
	@echo "\n=== Running Original Benchmark ==="
	./$(ORIG_TARGET) | tee $(ORIG_RESULTS)/run.log

run-optimized: optimized
	@echo "\n=== Running Optimized Tensor Core Benchmark ==="
	./$(OPT_TARGET) | tee $(OPT_RESULTS)/run.log

run-cublas: cublas
	@echo "\n=== Running cuBLAS Hardware Ceiling Benchmark ==="
	./$(CUBLAS_TARGET) | tee $(CUBLAS_RESULTS)/run.log

run-cutlass: cutlass
	@echo "\n=== Running CUTLASS Structured Baseline Benchmark ==="
	./$(CUTLASS_TARGET) | tee $(CUTLASS_RESULTS)/run.log

# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
#  plot         — all CSVs
#  bench        — build + run optimized, then plot
#  bench-all    — build + run all four, then plot
#  compare      — run everything then plot
# ─────────────────────────────────────────────────────────────────────────────
plot:
	python3 plot_results.py

bench: run-optimized plot

bench-all: run-original run-optimized run-cutlass run-cublas plot

compare: run plot

# ─────────────────────────────────────────────────────────────────────────────
#  Profiling
#
#  nsys targets   : nsys-original  nsys-optimized  nsys-cublas  nsys-cutlass
#                   nsys-all
#  ncu  targets   : ncu-original   ncu-optimized   ncu-cublas   ncu-cutlass
#                   ncu-all
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: nsys nsys-original nsys-optimized nsys-cublas nsys-cutlass nsys-all \
        ncu  ncu-original  ncu-optimized  ncu-cublas  ncu-cutlass  ncu-all

# ── nsys ─────────────────────────────────────────────────────────────────────
NSYS_FLAGS := --trace=cuda,nvtx

nsys: nsys-optimized   # keep old default behaviour

nsys-original: original
	@echo "\n=== nsys: original ==="
	nsys profile $(NSYS_FLAGS) \
	    --output=$(ORIG_RESULTS)/nsys_report_original \
	    ./$(ORIG_TARGET)

nsys-optimized: optimized
	@echo "\n=== nsys: optimized ==="
	nsys profile $(NSYS_FLAGS) \
	    --output=$(OPT_RESULTS)/nsys_report_optimized \
	    ./$(OPT_TARGET)

nsys-cublas: cublas
	@echo "\n=== nsys: cublas ==="
	nsys profile $(NSYS_FLAGS) \
	    --output=$(CUBLAS_RESULTS)/nsys_report_cublas \
	    ./$(CUBLAS_TARGET)

nsys-cutlass: cutlass
	@echo "\n=== nsys: cutlass ==="
	nsys profile $(NSYS_FLAGS) \
	    --output=$(CUTLASS_RESULTS)/nsys_report_cutlass \
	    ./$(CUTLASS_TARGET)

nsys-all: nsys-original nsys-optimized nsys-cublas nsys-cutlass

# ── ncu ──────────────────────────────────────────────────────────────────────
NCU_FLAGS := --set full --target-processes all -f
NCU      := sudo /usr/local/cuda-12.6/bin/ncu

ncu: ncu-optimized     # keep old default behaviour

ncu-original: original
	@echo "\n=== ncu: original ==="
	$(NCU) $(NCU_FLAGS) \
	    --export $(ORIG_RESULTS)/ncu_report_full_original \
	    ./$(ORIG_TARGET)

ncu-optimized: optimized
	@echo "\n=== ncu: optimized ==="
	$(NCU) $(NCU_FLAGS) \
	    --export $(OPT_RESULTS)/ncu_report_full_optimized \
	    ./$(OPT_TARGET)

ncu-cublas: cublas
	@echo "\n=== ncu: cublas ==="
	$(NCU) $(NCU_FLAGS) \
	    --export $(CUBLAS_RESULTS)/ncu_report_full_cublas \
	    ./$(CUBLAS_TARGET)

ncu-cutlass: cutlass
	@echo "\n=== ncu: cutlass ==="
	$(NCU) $(NCU_FLAGS) \
	    --export $(CUTLASS_RESULTS)/ncu_report_full_cutlass \
	    ./$(CUTLASS_TARGET)
ncu-all: ncu-original ncu-optimized ncu-cublas ncu-cutlass
# ─────────────────────────────────────────────────────────────────────────────
#  Clean
# ─────────────────────────────────────────────────────────────────────────────
clean: clean-original clean-optimized clean-cublas clean-cutlass

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

clean-cublas:
	rm -f $(CUBLAS_TARGET)
	rm -f $(CUBLAS_RESULTS)/benchmark_results.csv
	rm -f $(CUBLAS_RESULTS)/run.log
	rm -f $(CUBLAS_RESULTS)/*.png

clean-cutlass:
	rm -f $(CUTLASS_TARGET)
	rm -f $(CUTLASS_RESULTS)/benchmark_results.csv
	rm -f $(CUTLASS_RESULTS)/run.log
	rm -f $(CUTLASS_RESULTS)/*.png