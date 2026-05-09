#!/usr/bin/env python3
"""
plot_results.py  —  Unified Comparative Analysis Engine
Run from the repo root:
    python3 plot_results.py                          # all four CSVs
    python3 plot_results.py --baseline-only
    python3 plot_results.py --optimized-only
    python3 plot_results.py --cublas-only
    python3 plot_results.py --cutlass-only

CSVs read:
    baseline/results/benchmark_results.csv
    Optimized/results/benchmark_results.csv
    cublass_and_cutlass/results/benchmark_results_cublas.csv
    cublass_and_cutlass/results/benchmark_results_cutlass.csv

Plots saved alongside each CSV.
"""

import sys, os, csv, argparse
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Palette — all kernels across all four builds ─────────────────────────────
PALETTE = {
    # memory
    "Sequential":              "#2196F3",
    "Strided (x4)":            "#FF9800",
    "Random access":           "#F44336",
    # fp32 hand-written ladder
    "naive":                   "#BBDEFB",
    "tiled":                   "#64B5F6",
    "tiled_v2":                "#1976D2",
    "tiled_v3":                "#0D47A1",
    # hand-written Tensor Core
    "tc_basic":                "#FFB300",
    "tc_optimized":            "#E65100",
    # cuBLAS ceilings
    "cublas_sgemm":            "#AB47BC",
    "cublas_gemmex_fp16":      "#6A1B9A",
    # CUTLASS structured baselines
    "cutlass_sgemm":           "#26A69A",
    "cutlass_gemm_fp16_tc":    "#00695C",
    # alternate CUTLASS variant names the benchmark may emit
    "cutlass_fp32":            "#26A69A",
    "cutlass_fp16_tc":         "#00695C",
    "cutlass_fp16":            "#00897B",
}

LABEL = {
    "naive":                   "Naive\n(fp32, global)",
    "tiled":                   "Tiled v1\n(fp32, smem)",
    "tiled_v2":                "Tiled v2\n(WPT=4, 8×8)",
    "tiled_v3":                "Tiled v3\n(cp.async\n+WPT=4)",
    "tc_basic":                "TC Basic\n(fp16, 64×64)",
    "tc_optimized":            "TC Opt\n(fp16, 128×128\ndbl-buf)",
    "cublas_sgemm":            "cuBLAS\nSgemm\n(fp32)",
    "cublas_gemmex_fp16":      "cuBLAS\nGemmEx\n(fp16 TC)",
    "cutlass_sgemm":           "CUTLASS\nSgemm\n(fp32, Simt)",
    "cutlass_gemm_fp16_tc":    "CUTLASS\nGemmFp16\n(TC)",
    "cutlass_fp32":            "CUTLASS\nfp32\n(Simt)",
    "cutlass_fp16_tc":         "CUTLASS\nfp16\n(TC)",
    "cutlass_fp16":            "CUTLASS\nfp16\n(TC)",
}

# Canonical display order — scripts only plot variants present in the CSV
KERNEL_ORDER = [
    "naive",
    "tiled",
    "tiled_v2",
    "tiled_v3",
    "tc_basic",
    "tc_optimized",
    "cutlass_sgemm",
    "cutlass_fp32",
    "cutlass_gemm_fp16_tc",
    "cutlass_fp16_tc",
    "cutlass_fp16",
    "cublas_sgemm",
    "cublas_gemmex_fp16",
]

# Which variants belong to library baselines (used for annotation / shading)
LIBRARY_VARIANTS = {
    "cublas_sgemm", "cublas_gemmex_fp16",
    "cutlass_sgemm", "cutlass_gemm_fp16_tc",
    "cutlass_fp32", "cutlass_fp16_tc", "cutlass_fp16",
}
TC_VARIANTS = {
    "tc_basic", "tc_optimized",
    "cublas_gemmex_fp16", "cutlass_gemm_fp16_tc",
    "cutlass_fp16_tc", "cutlass_fp16",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def mb(n_elem):
    v = n_elem * 4 / (1 << 20)
    return f"{v*1024:.0f} KB" if v < 1 else f"{v:.0f} MB"

def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["n_or_size"]    = int(row["n_or_size"])
            row["avg_ms"]       = float(row["avg_ms"])
            row["metric_value"] = float(row["metric_value"])
            data[row["benchmark"]].append(row)
    return data

# ─── Plot 1: memory bandwidth ─────────────────────────────────────────────────
def plot_memory_bandwidth(rows, out_dir, tag):
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    for var, rlist in sorted(by_var.items()):
        rlist.sort(key=lambda r: r["n_or_size"])
        ax.plot([mb(r["n_or_size"]) for r in rlist],
                [r["metric_value"]  for r in rlist],
                marker="o", label=var,
                color=PALETTE.get(var, "#999"), linewidth=2)

    ax.set_title(f"Memory Bandwidth vs Array Size — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Array size (fp32)", fontsize=11)
    ax.set_ylabel("Bandwidth (GB/s)",  fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = os.path.join(out_dir, "memory_bandwidth.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 2: slowdown heatmap ─────────────────────────────────────────────────
def plot_slowdown_heatmap(rows, out_dir, tag):
    sizes   = sorted(set(r["n_or_size"] for r in rows))
    by_size = defaultdict(dict)
    for r in rows:
        by_size[r["n_or_size"]][r["variant"]] = r["avg_ms"]

    cmp_vars  = ["Strided (x4)", "Random access"]
    slowdowns = np.zeros((len(cmp_vars), len(sizes)))
    for j, sz in enumerate(sizes):
        base = by_size[sz].get("Sequential")
        for i, v in enumerate(cmp_vars):
            val = by_size[sz].get(v)
            slowdowns[i, j] = val / base if (base and val) else float("nan")

    fig, ax = plt.subplots(figsize=(9, 3))
    im = ax.imshow(slowdowns, cmap="YlOrRd", aspect="auto", vmin=1.0)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([mb(s) for s in sizes], rotation=30, ha="right")
    ax.set_yticks(range(len(cmp_vars)))
    ax.set_yticklabels(cmp_vars)
    ax.set_title(f"Slowdown vs Sequential  [{tag}]",
                 fontsize=12, fontweight="bold")
    for i in range(len(cmp_vars)):
        for j in range(len(sizes)):
            v = slowdowns[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}×", ha="center", va="center",
                        fontsize=9, color="black" if v < 5 else "white")
    plt.colorbar(im, ax=ax, label="Slowdown factor")
    plt.tight_layout()
    p = os.path.join(out_dir, "memory_slowdown_heatmap.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 3: GFLOPS ───────────────────────────────────────────────────────────
def plot_matmul_gflops(rows, out_dir, tag, file_tag=""):
    rb      = {r["variant"]: r for r in rows}
    present = [v for v in KERNEL_ORDER if v in rb]
    if not present:
        print(f"  [SKIP] matmul GFLOPS — no recognised variants in {tag} CSV")
        return
    gflops  = [rb[v]["metric_value"] for v in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 1.8), 6))
    bars = ax.bar([LABEL.get(v, v) for v in present], gflops,
                  color=[PALETTE.get(v, "#999") for v in present],
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(gflops) * 0.01,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Shade Tensor Core region
    tc_idx = [i for i, v in enumerate(present) if v in TC_VARIANTS]
    if tc_idx:
        ax.axvspan(bars[tc_idx[0]].get_x() - 0.1,
                   bars[tc_idx[-1]].get_x() + bars[tc_idx[-1]].get_width() + 0.1,
                   alpha=0.06, color="#E65100", label="Tensor Core kernels")

    # Shade library-baseline region
    lib_idx = [i for i, v in enumerate(present) if v in LIBRARY_VARIANTS]
    if lib_idx:
        ax.axvspan(bars[lib_idx[0]].get_x() - 0.1,
                   bars[lib_idx[-1]].get_x() + bars[lib_idx[-1]].get_width() + 0.1,
                   alpha=0.06, color="#6A1B9A", label="Library baselines")

    ax.set_title(f"Matmul GFLOPS — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_ylim(0, max(gflops) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, f"matmul_gflops{file_tag}.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 4: latency ──────────────────────────────────────────────────────────
def plot_matmul_latency(rows, out_dir, tag, file_tag=""):
    rb      = {r["variant"]: r for r in rows}
    present = [v for v in KERNEL_ORDER if v in rb]
    if not present:
        print(f"  [SKIP] matmul latency — no recognised variants in {tag} CSV")
        return
    latency = [rb[v]["avg_ms"] for v in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 1.8), 6))
    bars = ax.bar([LABEL.get(v, v) for v in present], latency,
                  color=[PALETTE.get(v, "#999") for v in present],
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(latency) * 0.01,
                f"{val:.1f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title(f"Matmul Latency — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg latency (ms)", fontsize=11)
    ax.set_ylim(0, max(latency) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"matmul_latency{file_tag}.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 5: speedup ladder ───────────────────────────────────────────────────
def plot_speedup_ladder(rows, out_dir, tag):
    rb = {r["variant"]: r for r in rows}
    if "naive" not in rb:
        print(f"  [SKIP] speedup ladder — 'naive' not in {tag} CSV")
        return
    base    = rb["naive"]["avg_ms"]
    present = [v for v in KERNEL_ORDER if v in rb]
    if not present:
        print(f"  [SKIP] speedup ladder — no recognised variants in {tag} CSV")
        return
    speedup = [base / rb[v]["avg_ms"] for v in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 1.8), 6))
    bars = ax.bar([LABEL.get(v, v) for v in present], speedup,
                  color=[PALETTE.get(v, "#999") for v in present],
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(speedup) * 0.01,
                f"{val:.2f}×", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1,
               label="Naive baseline (1×)")

    # Shade library region so it's visually distinct
    lib_idx = [i for i, v in enumerate(present) if v in LIBRARY_VARIANTS]
    if lib_idx:
        ax.axvspan(bars[lib_idx[0]].get_x() - 0.1,
                   bars[lib_idx[-1]].get_x() + bars[lib_idx[-1]].get_width() + 0.1,
                   alpha=0.07, color="#6A1B9A", label="Library baselines")

    ax.set_title(f"Speedup over Naive fp32 — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_ylim(0, max(speedup) * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "matmul_speedup.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 6: combined comparison (all four CSVs on one chart) ─────────────────
def plot_combined_gflops(all_matmul_rows, out_dir):
    """
    Draws a single grouped bar chart merging matmul rows from all CSVs.
    all_matmul_rows: list of (tag, rows) tuples.
    Only variants in KERNEL_ORDER are shown; order is preserved.
    """
    # Collect all variants across all tags (last-write wins per variant)
    combined = {}  # variant -> (tag, gflops)
    for tag, rows in all_matmul_rows:
        for r in rows:
            v = r["variant"]
            if v in KERNEL_ORDER:
                combined[v] = (tag, r["metric_value"])

    present = [v for v in KERNEL_ORDER if v in combined]
    if not present:
        print("  [SKIP] combined GFLOPS — no recognised variants across all CSVs")
        return

    gflops = [combined[v][1] for v in present]
    labels = [LABEL.get(v, v) for v in present]
    colors = [PALETTE.get(v, "#999") for v in present]

    fig, ax = plt.subplots(figsize=(max(9, len(present) * 1.9), 6))
    bars = ax.bar(labels, gflops, color=colors,
                  width=0.6, edgecolor="white", linewidth=1.2)
    for bar, val, v in zip(bars, gflops, present):
        suffix = "\n(lib)" if v in LIBRARY_VARIANTS else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(gflops) * 0.01,
                f"{val:.1f}{suffix}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    lib_idx = [i for i, v in enumerate(present) if v in LIBRARY_VARIANTS]
    if lib_idx:
        ax.axvspan(bars[lib_idx[0]].get_x() - 0.15,
                   bars[lib_idx[-1]].get_x() + bars[lib_idx[-1]].get_width() + 0.15,
                   alpha=0.07, color="#6A1B9A", label="Library baselines (cuBLAS / CUTLASS)")

    tc_idx = [i for i, v in enumerate(present) if v in TC_VARIANTS]
    if tc_idx:
        ax.axvspan(bars[tc_idx[0]].get_x() - 0.15,
                   bars[tc_idx[-1]].get_x() + bars[tc_idx[-1]].get_width() + 0.15,
                   alpha=0.05, color="#E65100", label="Tensor Core variants")

    ax.set_title("Full GEMM Ladder — Hand-written vs CUTLASS vs cuBLAS\n"
                 "Jetson Orin Nano (sm_87)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_ylim(0, max(gflops) * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "matmul_gflops_combined.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")


def plot_combined_latency(all_matmul_rows, out_dir):
    """
    Draws a single bar chart of avg_ms for all matmul variants across all CSVs.
    Lower is better — complements the GFLOPS combined chart.
    """
    combined = {}  # variant -> (tag, avg_ms)
    for tag, rows in all_matmul_rows:
        for r in rows:
            v = r["variant"]
            if v in KERNEL_ORDER:
                combined[v] = (tag, r["avg_ms"])

    present = [v for v in KERNEL_ORDER if v in combined]
    if not present:
        print("  [SKIP] combined latency -- no recognised variants across all CSVs")
        return

    latency = [combined[v][1] for v in present]
    labels  = [LABEL.get(v, v) for v in present]
    colors  = [PALETTE.get(v, "#999") for v in present]

    fig, ax = plt.subplots(figsize=(max(9, len(present) * 1.9), 6))
    bars = ax.bar(labels, latency, color=colors,
                  width=0.6, edgecolor="white", linewidth=1.2)
    for bar, val, v in zip(bars, latency, present):
        suffix = "\n(lib)" if v in LIBRARY_VARIANTS else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(latency) * 0.01,
                f"{val:.1f} ms{suffix}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    lib_idx = [i for i, v in enumerate(present) if v in LIBRARY_VARIANTS]
    if lib_idx:
        ax.axvspan(bars[lib_idx[0]].get_x() - 0.15,
                   bars[lib_idx[-1]].get_x() + bars[lib_idx[-1]].get_width() + 0.15,
                   alpha=0.07, color="#6A1B9A", label="Library baselines (cuBLAS / CUTLASS)")

    tc_idx = [i for i, v in enumerate(present) if v in TC_VARIANTS]
    if tc_idx:
        ax.axvspan(bars[tc_idx[0]].get_x() - 0.15,
                   bars[tc_idx[-1]].get_x() + bars[tc_idx[-1]].get_width() + 0.15,
                   alpha=0.05, color="#E65100", label="Tensor Core variants")

    ax.set_title("Full GEMM Ladder — Latency (lower is better)\n"
                 "Hand-written vs CUTLASS vs cuBLAS  |  Jetson Orin Nano (sm_87)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg latency (ms)", fontsize=11)
    ax.set_ylim(0, max(latency) * 1.28)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "matmul_latency_combined.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Text summary ─────────────────────────────────────────────────────────────
def print_text_summary(data, tag):
    print("\n" + "=" * 70)
    print(f"  {tag} — text summary")
    print("=" * 70)

    if "memory" in data:
        print("\n  Memory Access")
        print(f"  {'Variant':<22}  {'Size':>10}  {'BW (GB/s)':>10}")
        print("  " + "-" * 46)
        for r in sorted(data["memory"],
                        key=lambda r: (r["variant"], r["n_or_size"])):
            print(f"  {r['variant']:<22}  {mb(r['n_or_size']):>10}"
                  f"  {r['metric_value']:>10.2f}")

    if "matmul" in data:
        rb   = {r["variant"]: r for r in data["matmul"]}
        base = rb["naive"]["avg_ms"] if "naive" in rb else None
        print("\n  Matrix Multiply  (2048×2048)")
        print(f"  {'Kernel':<28}  {'ms':>8}  {'GFLOPS':>10}  {'vs naive':>10}  note")
        print("  " + "-" * 72)
        for v in KERNEL_ORDER:
            if v in rb:
                r   = rb[v]
                sp  = f"{base / r['avg_ms']:>8.2f}×" if base else "        —"
                note = ""
                if v in LIBRARY_VARIANTS:
                    note = "  ← library"
                elif v in TC_VARIANTS:
                    note = "  ← TC"
                print(f"  {v:<28}  {r['avg_ms']:>8.3f}  "
                      f"{r['metric_value']:>10.2f}  {sp}{note}")
        # Also print any variants in the CSV that aren't in KERNEL_ORDER
        unknown = [r for r in data["matmul"] if r["variant"] not in KERNEL_ORDER]
        for r in unknown:
            sp  = f"{base / r['avg_ms']:>8.2f}×" if base else "        —"
            print(f"  {r['variant']:<28}  {r['avg_ms']:>8.3f}  "
                  f"{r['metric_value']:>10.2f}  {sp}  ← (unrecognised variant)")
    print()

# ─── Per-CSV runner ───────────────────────────────────────────────────────────
def process(csv_path, tag, out_dir=None):
    if not os.path.exists(csv_path):
        print(f"[SKIP] Not found: {csv_path}")
        return None

    resolved_out = out_dir or os.path.dirname(os.path.abspath(csv_path))
    # Derive suffix from CSV filename so outputs never collide:
    # "benchmark_results_cublas.csv"  -> file_tag = "_cublas"
    # "benchmark_results_cutlass.csv" -> file_tag = "_cutlass"
    # "benchmark_results.csv"         -> file_tag = ""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    prefix = "benchmark_results"
    file_tag = stem[len(prefix):] if stem.startswith(prefix) else f"_{stem}"
    print(f"\nLoading [{tag}]: {csv_path}")
    data = load_csv(csv_path)
    print_text_summary(data, tag)

    if not HAS_MPL:
        print("[WARN] matplotlib/numpy not found — skipping plots.")
        return data

    print(f"Generating plots → {resolved_out}/")
    if "memory" in data:
        plot_memory_bandwidth(data["memory"], resolved_out, tag)
        plot_slowdown_heatmap(data["memory"], resolved_out, tag)
    if "matmul" in data:
        plot_matmul_gflops  (data["matmul"], resolved_out, tag, file_tag)
        plot_matmul_latency (data["matmul"], resolved_out, tag, file_tag)
        plot_speedup_ladder (data["matmul"], resolved_out, tag)

    return data

# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot baseline, optimized, cuBLAS, and CUTLASS benchmark results.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline-only",  action="store_true")
    group.add_argument("--optimized-only", action="store_true")
    group.add_argument("--cublas-only",    action="store_true")
    group.add_argument("--cutlass-only",   action="store_true")
    args = parser.parse_args()

    run_baseline  = not (args.optimized_only or args.cublas_only or args.cutlass_only)
    run_optimized = not (args.baseline_only  or args.cublas_only or args.cutlass_only)
    run_cublas    = not (args.baseline_only  or args.optimized_only or args.cutlass_only)
    run_cutlass   = not (args.baseline_only  or args.optimized_only or args.cublas_only)

    # Results dir for combined chart (root-level results/)
    combined_dir = "results"
    os.makedirs(combined_dir, exist_ok=True)

    all_matmul = []   # accumulate for combined chart

    if run_baseline:
        d = process("baseline/results/benchmark_results.csv", "Baseline")
        if d and "matmul" in d:
            all_matmul.append(("Baseline", d["matmul"]))

    if run_optimized:
        d = process("Optimized/results/benchmark_results.csv", "Optimized")
        if d and "matmul" in d:
            all_matmul.append(("Optimized", d["matmul"]))

    if run_cublas:
        d = process(
            "cublass_and_cutlass/results/benchmark_results_cublas.csv",
            "cuBLAS ceiling",
            out_dir="cublass_and_cutlass/results",
        )
        if d and "matmul" in d:
            all_matmul.append(("cuBLAS", d["matmul"]))

    if run_cutlass:
        d = process(
            "cublass_and_cutlass/results/benchmark_results_cutlass.csv",
            "CUTLASS baseline",
            out_dir="cublass_and_cutlass/results",
        )
        if d and "matmul" in d:
            all_matmul.append(("CUTLASS", d["matmul"]))

    # Combined chart — only when more than one source has matmul data
    if HAS_MPL and len(all_matmul) > 1:
        print(f"\nGenerating combined chart → {combined_dir}/")
        plot_combined_gflops(all_matmul, combined_dir)
        plot_combined_latency(all_matmul, combined_dir)

if __name__ == "__main__":
    main()