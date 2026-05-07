#!/usr/bin/env python3
"""
scripts/plot_results.py
Phase 5 — Comparative Analysis Engine

Reads  results/benchmark_results.csv  and generates:
  1. memory_bandwidth.png          — BW vs array size, all 3 access patterns
  2. memory_slowdown_heatmap.png   — strided / random slowdown vs sequential
  3. matmul_gflops.png             — GFLOPS bar chart (4 kernels)
  4. matmul_latency.png            — latency bar chart (4 kernels)
  5. matmul_speedup.png            — speedup ladder over naive

Usage:
    python3 scripts/plot_results.py                    # default CSV path
    python3 scripts/plot_results.py <path/to/csv>
"""

import sys, os, csv
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─────────────────────────────────────────────────────────────────────────────
#  Colours & labels
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "Sequential":    "#2196F3",
    "Strided (x4)":  "#FF9800",
    "Random access": "#F44336",
    "naive":         "#BBDEFB",
    "tiled":         "#42A5F5",
    "tc_basic":      "#1565C0",
    "tc_optimized":  "#0D2E6E",
}

LABEL = {
    "naive":         "Naive\n(fp32, global)",
    "tiled":         "Tiled\n(fp32, shared)",
    "tc_basic":      "TC Basic\n(fp16, 64×64)",
    "tc_optimized":  "TC Optimized\n(fp16, 128×128\ndbl-buf)",
}

KERNEL_ORDER = ["naive", "tiled", "tc_basic", "tc_optimized"]

def mb(n_elem):
    v = n_elem * 4 / (1 << 20)
    return f"{v*1024:.0f} KB" if v < 1 else f"{v:.0f} MB"

# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["n_or_size"]    = int(row["n_or_size"])
            row["avg_ms"]       = float(row["avg_ms"])
            row["metric_value"] = float(row["metric_value"])
            data[row["benchmark"]].append(row)
    return data

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1 — Memory bandwidth sweep
# ─────────────────────────────────────────────────────────────────────────────
def plot_memory_bandwidth(rows, out_dir):
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    for var, rlist in sorted(by_var.items()):
        rlist.sort(key=lambda r: r["n_or_size"])
        xs  = [mb(r["n_or_size"])    for r in rlist]
        bws = [r["metric_value"]     for r in rlist]
        ax.plot(xs, bws, marker="o", label=var,
                color=PALETTE.get(var, "#999"), linewidth=2)

    ax.set_title("Memory Bandwidth vs Array Size — Jetson Orin Nano",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Array size (fp32)",  fontsize=11)
    ax.set_ylabel("Bandwidth (GB/s)",   fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p = os.path.join(out_dir, "memory_bandwidth.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 2 — Slowdown heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_slowdown_heatmap(rows, out_dir):
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
    ax.set_title("Slowdown vs Sequential (higher = worse)",
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
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3 — Matmul GFLOPS
# ─────────────────────────────────────────────────────────────────────────────
def plot_matmul_gflops(rows, out_dir):
    rb = {r["variant"]: r for r in rows}
    present = [v for v in KERNEL_ORDER if v in rb]
    labels  = [LABEL[v]            for v in present]
    gflops  = [rb[v]["metric_value"] for v in present]
    colors  = [PALETTE[v]          for v in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, gflops, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(gflops)*0.01,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_title("Matmul GFLOPS — Jetson Orin Nano (2048×2048)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("GFLOPS", fontsize=11)
    ax.set_ylim(0, max(gflops) * 1.20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "matmul_gflops.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4 — Matmul latency
# ─────────────────────────────────────────────────────────────────────────────
def plot_matmul_latency(rows, out_dir):
    rb = {r["variant"]: r for r in rows}
    present = [v for v in KERNEL_ORDER if v in rb]
    labels  = [LABEL[v]         for v in present]
    latency = [rb[v]["avg_ms"]  for v in present]
    colors  = [PALETTE[v]       for v in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, latency, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(latency)*0.01,
                f"{val:.1f} ms", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_title("Matmul Latency — Jetson Orin Nano (2048×2048)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg latency (ms)", fontsize=11)
    ax.set_ylim(0, max(latency) * 1.20)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "matmul_latency.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
#  Plot 5 — Speedup ladder over naive
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup_ladder(rows, out_dir):
    rb = {r["variant"]: r for r in rows}
    if "naive" not in rb:
        return
    base    = rb["naive"]["avg_ms"]
    present = [v for v in KERNEL_ORDER if v in rb]
    labels  = [LABEL[v]                      for v in present]
    speedup = [base / rb[v]["avg_ms"]        for v in present]
    colors  = [PALETTE[v]                    for v in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, speedup, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(speedup)*0.01,
                f"{val:.2f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Speedup over Naive — Jetson Orin Nano (2048×2048)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup (×)", fontsize=11)
    ax.set_ylim(0, max(speedup) * 1.22)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "matmul_speedup.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
#  Text summary (always, no matplotlib needed)
# ─────────────────────────────────────────────────────────────────────────────
def print_text_summary(data):
    print("\n" + "="*64)
    print("  Phase 5 — Comparative Analysis (text)")
    print("="*64)

    if "memory" in data:
        print("\n  Memory Access")
        print(f"  {'Variant':<22}  {'Size':>10}  {'BW (GB/s)':>10}")
        print("  " + "-"*46)
        for r in sorted(data["memory"],
                        key=lambda r: (r["variant"], r["n_or_size"])):
            print(f"  {r['variant']:<22}  {mb(r['n_or_size']):>10}  "
                  f"{r['metric_value']:>10.2f}")

    if "matmul" in data:
        rb = {r["variant"]: r for r in data["matmul"]}
        print("\n  Matrix Multiply")
        print(f"  {'Kernel':<22}  {'ms':>8}  {'GFLOPS':>10}")
        print("  " + "-"*46)
        for v in KERNEL_ORDER:
            if v in rb:
                r = rb[v]
                print(f"  {v:<22}  {r['avg_ms']:>8.3f}  "
                      f"{r['metric_value']:>10.2f}")
        if "naive" in rb:
            base = rb["naive"]["avg_ms"]
            print()
            for v in KERNEL_ORDER[1:]:
                if v in rb:
                    print(f"  {v:<22}  speedup: {base/rb[v]['avg_ms']:.2f}×")
    print()

# ─────────────────────────────────────────────────────────────────────────────
def main():
    path    = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_results.csv"
    out_dir = os.path.dirname(os.path.abspath(path))

    if not os.path.exists(path):
        print(f"[ERROR] Not found: {path}")
        print("  Run ./cuda_bench first.")
        sys.exit(1)

    print(f"Loading: {path}")
    data = load_csv(path)
    print_text_summary(data)

    if not HAS_MPL:
        print("[WARN] matplotlib/numpy not found — skipping plots.")
        print("  pip3 install matplotlib numpy")
        return

    print("Generating plots …")
    if "memory" in data:
        plot_memory_bandwidth   (data["memory"], out_dir)
        plot_slowdown_heatmap   (data["memory"], out_dir)
    if "matmul" in data:
        plot_matmul_gflops      (data["matmul"], out_dir)
        plot_matmul_latency     (data["matmul"], out_dir)
        plot_speedup_ladder     (data["matmul"], out_dir)

    print(f"\nAll plots saved to: {out_dir}/")

if __name__ == "__main__":
    main()
