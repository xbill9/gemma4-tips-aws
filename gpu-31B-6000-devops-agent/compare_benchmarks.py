import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def generate_comparison():
    gpu_csv = "/home/xbill/gemma4-tips/gpu-31B-6000-devops-agent/matrix_benchmark_results.csv"
    tpu_csv = "/home/xbill/gemma4-tips/tpu-31B-devops-agent/matrix_benchmark_results.csv"

    if not os.path.exists(gpu_csv) or not os.path.exists(tpu_csv):
        print("Error: Missing CSV files.")
        return

    gpu_df = pd.read_csv(gpu_csv)
    tpu_df = pd.read_csv(tpu_csv)

    # Set up matplotlib style
    plt.rcParams["figure.facecolor"] = "#0f172a"  # Slate 900
    plt.rcParams["axes.facecolor"] = "#1e293b"  # Slate 800
    plt.rcParams["text.color"] = "#f8fafc"  # Slate 50
    plt.rcParams["axes.labelcolor"] = "#cbd5e1"  # Slate 300
    plt.rcParams["xtick.color"] = "#94a3b8"  # Slate 400
    plt.rcParams["ytick.color"] = "#94a3b8"  # Slate 400
    plt.rcParams["grid.color"] = "#334155"  # Slate 700
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f172a")
    fig.suptitle(
        "Gemma 4 (31B) serving: RTX 6000 Pro GPU vs TPU v6e-4", fontsize=18, fontweight="bold", color="#f8fafc", y=0.96
    )

    # We will compare a small context (8 tokens) and a large context (16,384 tokens)
    sizes_to_plot = [8, 16384]

    # Subplot 1: Throughput (Tokens/s) vs Concurrency
    ax1 = axes[0]
    ax1.set_title("Token Throughput (tokens/sec) vs Concurrency", fontsize=13, fontweight="semibold", pad=15)
    ax1.grid(True)

    # Subplot 2: Average Latency vs Concurrency
    ax2 = axes[1]
    ax2.set_title("Average Latency (seconds) vs Concurrency", fontsize=13, fontweight="semibold", pad=15)
    ax2.grid(True)

    # TPU colors: Greens/Teals
    # GPU colors: Blues/Pinks
    styles = {
        ("TPU", 8): {"color": "#10b981", "ls": "-", "marker": "s", "label": "TPU v6e (8 ctx)"},
        ("TPU", 16384): {"color": "#059669", "ls": "--", "marker": "D", "label": "TPU v6e (16k ctx)"},
        ("GPU", 8): {"color": "#38bdf8", "ls": "-", "marker": "o", "label": "RTX 6000 GPU (8 ctx)"},
        ("GPU", 16384): {"color": "#ec4899", "ls": "--", "marker": "x", "label": "RTX 6000 GPU (16k ctx)"},
    }

    # Filter and plot TPU
    for size in sizes_to_plot:
        sub_tpu = tpu_df[tpu_df["context_size"] == size].sort_values("concurrency")
        sub_gpu = gpu_df[gpu_df["context_size"] == size].sort_values("concurrency")

        if not sub_tpu.empty:
            style = styles[("TPU", size)]
            ax1.plot(
                sub_tpu["concurrency"],
                sub_tpu["tokens_per_sec"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=2.5,
                label=style["label"],
            )
            ax2.plot(
                sub_tpu["concurrency"],
                sub_tpu["avg_latency"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=2.5,
                label=style["label"],
            )

        if not sub_gpu.empty:
            style = styles[("GPU", size)]
            ax1.plot(
                sub_gpu["concurrency"],
                sub_gpu["tokens_per_sec"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=2.5,
                label=style["label"],
            )
            ax2.plot(
                sub_gpu["concurrency"],
                sub_gpu["avg_latency"],
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                linewidth=2.5,
                label=style["label"],
            )

    ax1.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
    ax1.set_ylabel("Tokens per Second", labelpad=10)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax1.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    ax2.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
    ax2.set_ylabel("Average Latency (s)", labelpad=10)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax2.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    plt.tight_layout(rect=(0, 0.03, 1, 0.92))

    chart_path = "/home/xbill/gemma4-tips/gpu-31B-6000-devops-agent/gpu_tpu_comparison.png"
    plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Comparison chart saved to {chart_path}")


if __name__ == "__main__":
    generate_comparison()
