#!/usr/bin/env python3
"""
plot_results.py  —  Unified Comparative Analysis Engine
Run from the repo root:
    python3 plot_results.py                          # plots both baseline + optimized
    python3 plot_results.py --baseline-only
    python3 plot_results.py --optimized-only

CSVs read:
    baseline/results/benchmark_results.csv
    Optimized/results/benchmark_results.csv

Plots saved alongside each CSV:
    baseline/results/*.png
    Optimized/results/*.png
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

# ─── Palette (covers all kernels from both builds) ────────────────────────────
PALETTE = {
    # memory
    "Sequential":    "#2196F3",
    "Strided (x4)":  "#FF9800",
    "Random access": "#F44336",
    # fp32 matmul
    "naive":         "#BBDEFB",
    "tiled":         "#64B5F6",
    "tiled_v2":      "#1976D2",
    "tiled_v3":      "#0D47A1",
    # Tensor Core
    "tc_basic":      "#FFB300",
    "tc_optimized":  "#E65100",
}

LABEL = {
    "naive":         "Naive\n(fp32, global)",
    "tiled":         "Tiled v1\n(fp32, smem)",
    "tiled_v2":      "Tiled v2\n(WPT=4, 8×8)",
    "tiled_v3":      "Tiled v3\n(cp.async\n+WPT=4)",
    "tc_basic":      "TC Basic\n(fp16, 64×64)",
    "tc_optimized":  "TC Optimized\n(fp16, 128×128\ndbl-buf)",
}

# Full order — each script only plots kernels actually present in the CSV
KERNEL_ORDER = ["naive", "tiled", "tiled_v2", "tiled_v3", "tc_basic", "tc_optimized"]

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
    ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
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
    ax.set_yticks(range(len(cmp_vars))); ax.set_yticklabels(cmp_vars)
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
def plot_matmul_gflops(rows, out_dir, tag):
    rb      = {r["variant"]: r for r in rows}
    present = [v for v in KERNEL_ORDER if v in rb]
    gflops  = [rb[v]["metric_value"] for v in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present)*1.8), 6))
    bars = ax.bar([LABEL[v] for v in present], gflops,
                  color=[PALETTE[v] for v in present],
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(gflops)*0.01,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    tc_idx = [i for i, v in enumerate(present) if v.startswith("tc_")]
    if tc_idx:
        ax.axvspan(bars[tc_idx[0]].get_x() - 0.1,
                   bars[tc_idx[-1]].get_x() + bars[tc_idx[-1]].get_width() + 0.1,
                   alpha=0.08, color="#E65100", label="Tensor Core kernels")

    ax.set_title(f"Matmul GFLOPS — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_ylim(0, max(gflops) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9); plt.tight_layout()
    p = os.path.join(out_dir, "matmul_gflops.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 4: latency ──────────────────────────────────────────────────────────
def plot_matmul_latency(rows, out_dir, tag):
    rb      = {r["variant"]: r for r in rows}
    present = [v for v in KERNEL_ORDER if v in rb]
    latency = [rb[v]["avg_ms"] for v in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present)*1.8), 6))
    bars = ax.bar([LABEL[v] for v in present], latency,
                  color=[PALETTE[v] for v in present],
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(latency)*0.01,
                f"{val:.1f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_title(f"Matmul Latency — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg latency (ms)", fontsize=11)
    ax.set_ylim(0, max(latency) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.5); plt.tight_layout()
    p = os.path.join(out_dir, "matmul_latency.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Plot 5: speedup ladder ───────────────────────────────────────────────────
def plot_speedup_ladder(rows, out_dir, tag):
    rb = {r["variant"]: r for r in rows}
    if "naive" not in rb:
        return
    base    = rb["naive"]["avg_ms"]
    present = [v for v in KERNEL_ORDER if v in rb]
    speedup = [base / rb[v]["avg_ms"] for v in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present)*1.8), 6))
    bars = ax.bar([LABEL[v] for v in present], speedup,
                  color=[PALETTE[v] for v in present],
                  width=0.55, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(speedup)*0.01,
                f"{val:.2f}×", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="Naive baseline (1×)")
    ax.set_title(f"Speedup over Naive fp32 — Jetson Orin Nano  [{tag}]",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_ylim(0, max(speedup) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9); plt.tight_layout()
    p = os.path.join(out_dir, "matmul_speedup.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  Saved: {p}")

# ─── Text summary ─────────────────────────────────────────────────────────────
def print_text_summary(data, tag):
    print("\n" + "="*70)
    print(f"  {tag} — text summary")
    print("="*70)

    if "memory" in data:
        print("\n  Memory Access")
        print(f"  {'Variant':<22}  {'Size':>10}  {'BW (GB/s)':>10}")
        print("  " + "-"*46)
        for r in sorted(data["memory"], key=lambda r: (r["variant"], r["n_or_size"])):
            print(f"  {r['variant']:<22}  {mb(r['n_or_size']):>10}  {r['metric_value']:>10.2f}")

    if "matmul" in data:
        rb   = {r["variant"]: r for r in data["matmul"]}
        base = rb["naive"]["avg_ms"] if "naive" in rb else None
        print("\n  Matrix Multiply  (2048×2048)")
        print(f"  {'Kernel':<24}  {'ms':>8}  {'GFLOPS':>10}  {'vs naive':>10}")
        print("  " + "-"*58)
        for v in KERNEL_ORDER:
            if v in rb:
                r  = rb[v]
                sp = f"  {base/r['avg_ms']:.2f}×" if base else ""
                tc = "  ← TC" if v.startswith("tc_") else ""
                print(f"  {v:<24}  {r['avg_ms']:>8.3f}  {r['metric_value']:>10.2f}{sp}{tc}")
    print()

# ─── Per-CSV runner ───────────────────────────────────────────────────────────
def process(csv_path, tag):
    if not os.path.exists(csv_path):
        print(f"[SKIP] Not found: {csv_path}")
        return

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    print(f"\nLoading [{tag}]: {csv_path}")
    data = load_csv(csv_path)
    print_text_summary(data, tag)

    if not HAS_MPL:
        print("[WARN] matplotlib/numpy not found — skipping plots.")
        return

    print(f"Generating plots → {out_dir}/")
    if "memory" in data:
        plot_memory_bandwidth (data["memory"], out_dir, tag)
        plot_slowdown_heatmap (data["memory"], out_dir, tag)
    if "matmul" in data:
        plot_matmul_gflops    (data["matmul"], out_dir, tag)
        plot_matmul_latency   (data["matmul"], out_dir, tag)
        plot_speedup_ladder   (data["matmul"], out_dir, tag)

# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot baseline and/or optimized benchmark results.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline-only",  action="store_true", help="Plot baseline only")
    group.add_argument("--optimized-only", action="store_true", help="Plot optimized only")
    args = parser.parse_args()

    run_baseline  = not args.optimized_only
    run_optimized = not args.baseline_only

    if run_baseline:
        process("baseline/results/benchmark_results.csv", "Baseline")
    if run_optimized:
        process("Optimized/results/benchmark_results.csv", "Optimized")

if __name__ == "__main__":
    main()