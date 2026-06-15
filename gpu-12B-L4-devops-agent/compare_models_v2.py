import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def generate_charts():
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
        df["p95_latency"] = pd.to_numeric(df["p95_latency"])
        df["success_rate"] = pd.to_numeric(df["success_rate"])

    # Calculate token throughput for Standard model (requests/sec * max_tokens)
    # The standard sweep was run with max_tokens=1, so req_per_sec is equivalent to tokens_per_sec.
    std_df["tokens_per_sec"] = std_df["req_per_sec"] * 1.0
    if "tokens_per_sec" not in qat_df.columns:
        qat_df["tokens_per_sec"] = qat_df["req_per_sec"] * 1.0

    # Set up matplotlib style for rich aesthetics
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

    # Context window sizes and colors to use
    sizes_to_plot = [128, 1024, 8192]
    colors = {
        128: "#38bdf8",  # Sky blue
        1024: "#a855f7",  # Purple
        8192: "#f43f5e",  # Rose/Red
        16384: "#10b981",  # Emerald green (used in success rate plot)
    }
    concurrencies_to_1024 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # --- CHART 1: Throughput Comparison (Requests vs Tokens) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0f172a")
    fig.suptitle(
        "Gemma 4 (12B) Serving Throughput: Standard FP8 vs QAT INT4",
        fontsize=18,
        fontweight="bold",
        color="#f8fafc",
        y=0.96,
    )

    ax1.set_title(
        "Request Throughput (Requests/sec) vs Concurrency\n(Short Completions: Standard = 1 token, QAT = 16 tokens)",
        fontsize=13,
        pad=15,
    )
    ax1.grid(True)
    ax2.set_title("Equivalent Token Generation Throughput (tokens/sec) vs Concurrency", fontsize=13, pad=15)
    ax2.grid(True)

    for size in sizes_to_plot:
        sub_std = std_df[(std_df["context_size"] == size) & (std_df["concurrency"] <= 1024)].sort_values("concurrency")
        sub_qat = qat_df[(qat_df["context_size"] == size) & (qat_df["concurrency"] <= 1024)].sort_values("concurrency")
        color = colors[size]

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
                sub_std["tokens_per_sec"],
                color=color,
                linestyle="-",
                marker="o",
                linewidth=2.5,
                label=f"Std (FP8) - {size} ctx",
            )

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
                sub_qat["tokens_per_sec"],
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.0,
                label=f"QAT (INT4) - {size} ctx",
            )

    for ax in [ax1, ax2]:
        ax.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_xticks(concurrencies_to_1024)
        ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left")

    ax1.set_ylabel("Throughput (Req/s)", labelpad=10)
    ax2.set_ylabel("Throughput (Tokens/s)", labelpad=10)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig("/home/xbill/gemma4-tips/gpu-12B-L4-devops-agent/throughput_comparison.png", dpi=300)
    plt.close()

    # --- CHART 2: Latency Profiles (Average vs P95) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0f172a")
    fig.suptitle(
        "Gemma 4 (12B) Latency Profiles: Standard FP8 vs QAT INT4",
        fontsize=18,
        fontweight="bold",
        color="#f8fafc",
        y=0.96,
    )

    ax1.set_title("Average Latency (seconds) vs Concurrency", fontsize=13, pad=15)
    ax1.grid(True)
    ax2.set_title("95th Percentile (P95) Latency (seconds) vs Concurrency", fontsize=13, pad=15)
    ax2.grid(True)

    for size in sizes_to_plot:
        sub_std = std_df[(std_df["context_size"] == size) & (std_df["concurrency"] <= 1024)].sort_values("concurrency")
        sub_qat = qat_df[(qat_df["context_size"] == size) & (qat_df["concurrency"] <= 1024)].sort_values("concurrency")
        color = colors[size]

        if not sub_std.empty:
            ax1.plot(
                sub_std["concurrency"],
                sub_std["avg_latency"],
                color=color,
                linestyle="-",
                marker="o",
                linewidth=2.5,
                label=f"Std (FP8) - {size} ctx",
            )
            ax2.plot(
                sub_std["concurrency"],
                sub_std["p95_latency"],
                color=color,
                linestyle="-",
                marker="o",
                linewidth=2.5,
                label=f"Std (FP8) - {size} ctx",
            )

        if not sub_qat.empty:
            ax1.plot(
                sub_qat["concurrency"],
                sub_qat["avg_latency"],
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.0,
                label=f"QAT (INT4) - {size} ctx",
            )
            ax2.plot(
                sub_qat["concurrency"],
                sub_qat["p95_latency"],
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.0,
                label=f"QAT (INT4) - {size} ctx",
            )

    for ax in [ax1, ax2]:
        ax.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_xticks(concurrencies_to_1024)
        ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left")

    ax1.set_ylabel("Average Latency (s)", labelpad=10)
    ax2.set_ylabel("P95 Latency (s)", labelpad=10)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig("/home/xbill/gemma4-tips/gpu-12B-L4-devops-agent/latency_comparison.png", dpi=300)
    plt.close()

    # --- CHART 3: Success Rate Stability Matrix ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0f172a")
    fig.suptitle(
        "Gemma 4 (12B) Request Success Rate & Stability comparison",
        fontsize=18,
        fontweight="bold",
        color="#f8fafc",
        y=0.96,
    )

    ax1.set_title("Standard FP8 Success Rate vs Concurrency", fontsize=13, pad=15)
    ax1.grid(True)
    ax2.set_title("QAT INT4 Success Rate vs Concurrency", fontsize=13, pad=15)
    ax2.grid(True)

    sizes_for_success = [128, 1024, 8192, 16384]

    for size in sizes_for_success:
        sub_std = std_df[(std_df["context_size"] == size) & (std_df["concurrency"] <= 1024)].sort_values("concurrency")
        sub_qat = qat_df[(qat_df["context_size"] == size) & (qat_df["concurrency"] <= 1024)].sort_values("concurrency")
        color = colors[size]

        if not sub_std.empty:
            ax1.plot(
                sub_std["concurrency"],
                sub_std["success_rate"] * 100.0,
                color=color,
                linestyle="-",
                marker="o",
                linewidth=2.5,
                label=f"Context: {size}",
            )
        if not sub_qat.empty:
            ax2.plot(
                sub_qat["concurrency"],
                sub_qat["success_rate"] * 100.0,
                color=color,
                linestyle="--",
                marker="D",
                linewidth=2.5,
                label=f"Context: {size}",
            )

    for ax in [ax1, ax2]:
        ax.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax.set_xticks(concurrencies_to_1024)
        ax.set_ylabel("Success Rate (%)", labelpad=10)
        ax.set_ylim(-5, 105)
        ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="lower left")

    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    plt.savefig("/home/xbill/gemma4-tips/gpu-12B-L4-devops-agent/success_rate_comparison.png", dpi=300)
    plt.close()

    # --- CHART 4: Latency scaling vs Context size (fixed low concurrency) ---
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#0f172a")
    ax.set_title("Latency Scaling vs Input Context Size (Concurrency = 8)", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True)

    # Filter data for concurrency = 8
    std_c8 = std_df[std_df["concurrency"] == 8].sort_values("context_size")
    qat_c8 = qat_df[qat_df["concurrency"] == 8].sort_values("context_size")

    # Filter out context size 16384 for Standard since it fails
    std_c8 = std_c8[std_c8["context_size"] < 16384]

    ax.plot(
        std_c8["context_size"],
        std_c8["avg_latency"],
        color="#38bdf8",
        linestyle="-",
        marker="o",
        linewidth=3.0,
        label="Standard FP8 (16K Limit)",
    )
    ax.plot(
        qat_c8["context_size"],
        qat_c8["avg_latency"],
        color="#10b981",
        linestyle="--",
        marker="D",
        linewidth=2.5,
        label="QAT INT4 (32K Limit)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context Size (tokens)", labelpad=10)
    ax.set_ylabel("Average Latency (seconds)", labelpad=10)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    plt.tight_layout()
    plt.savefig("/home/xbill/gemma4-tips/gpu-12B-L4-devops-agent/context_scaling_comparison.png", dpi=300)
    plt.close()

    print("All comparison charts successfully generated.")


if __name__ == "__main__":
    generate_charts()
