import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def generate_comparison():
    standard_csv = "/home/xbill/gemma4-tips/gpu-12B-L4-devops-agent/benchmark_sweep_results.csv"
    qat_csv = "/home/xbill/gemma4-tips/gpu-12B-qat-L4-devops-agent/benchmark_sweep_results.csv"

    if not os.path.exists(standard_csv) or not os.path.exists(qat_csv):
        print(f"Error: Missing CSV files.\nStandard CSV: {standard_csv}\nQAT CSV: {qat_csv}")
        return

    std_df = pd.read_csv(standard_csv)
    qat_df = pd.read_csv(qat_csv)

    # Clean data (ensure numeric types)
    for df in [std_df, qat_df]:
        df["concurrency"] = pd.to_numeric(df["concurrency"])
        df["context_size"] = pd.to_numeric(df["context_size"])
        df["req_per_sec"] = pd.to_numeric(df["req_per_sec"])
        df["avg_latency"] = pd.to_numeric(df["avg_latency"])

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), facecolor="#0f172a")
    fig.suptitle(
        "Gemma 4 (12B) Model Serving Comparison on NVIDIA L4 GPU\nStandard (FP8 Weight, 16K Max Ctx) vs QAT (INT4 Weight, 32K Max Ctx)",
        fontsize=18,
        fontweight="bold",
        color="#f8fafc",
        y=0.96,
    )

    # Context window sizes to compare
    sizes_to_plot = [128, 1024, 8192]

    # Subplot 1: Throughput (Req/s) vs Concurrency
    ax1 = axes[0]
    ax1.set_title("Request Throughput (Req/s) vs Concurrency", fontsize=13, fontweight="semibold", pad=15)
    ax1.grid(True)

    # Subplot 2: Average Latency vs Concurrency
    ax2 = axes[1]
    ax2.set_title("Average Latency (seconds) vs Concurrency", fontsize=13, fontweight="semibold", pad=15)
    ax2.grid(True)

    # Colors for different context sizes
    colors = {
        128: "#38bdf8",  # Sky blue
        1024: "#a855f7",  # Purple
        8192: "#f43f5e",  # Rose/Red
    }

    # Plot lines
    for size in sizes_to_plot:
        # Filter data
        sub_std = std_df[std_df["context_size"] == size].sort_values("concurrency")
        sub_qat = qat_df[qat_df["context_size"] == size].sort_values("concurrency")

        # Limit to concurrencies <= 1024 to align Standard and QAT
        sub_std = sub_std[sub_std["concurrency"] <= 1024]
        sub_qat = sub_qat[sub_qat["concurrency"] <= 1024]

        color = colors[size]

        # Standard (Solid line, circles)
        if not sub_std.empty:
            ax1.plot(
                sub_std["concurrency"],
                sub_std["req_per_sec"],
                color=color,
                linestyle="-",
                marker="o",
                linewidth=2.5,
                label=f"Std (FP8) - {size} ctx",
            )
            ax2.plot(
                sub_std["concurrency"],
                sub_std["avg_latency"],
                color=color,
                linestyle="-",
                marker="o",
                linewidth=2.5,
                label=f"Std (FP8) - {size} ctx",
            )

        # QAT (Dashed line, diamonds)
        if not sub_qat.empty:
            ax1.plot(
                sub_qat["concurrency"],
                sub_qat["req_per_sec"],
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.0,
                label=f"QAT (INT4) - {size} ctx",
            )
            ax2.plot(
                sub_qat["concurrency"],
                sub_qat["avg_latency"],
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.0,
                label=f"QAT (INT4) - {size} ctx",
            )

    # Configure X axes
    for ax in [ax1, ax2]:
        ax.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        # Position legend at bottom right or top left
        ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left")

    ax1.set_ylabel("Throughput (Requests per Second)", labelpad=10)
    ax2.set_ylabel("Average Latency (s)", labelpad=10)

    plt.tight_layout(rect=(0, 0.03, 1, 0.90))

    chart_path = "/home/xbill/gemma4-tips/gpu-12B-L4-devops-agent/model_comparison_chart.png"
    plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Comparison chart saved to {chart_path}")


if __name__ == "__main__":
    generate_comparison()
