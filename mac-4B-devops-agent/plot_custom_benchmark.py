import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Set modern style settings for a high-end look
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


def generate_plots():
    # 1. Load data
    csv_path = "custom_benchmark_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Sort data for clean plotting
    df = df.sort_values(by=["concurrency", "max_tokens"])

    concurrencies = sorted(df["concurrency"].unique())

    # Colors for different concurrency levels (Vibrant Palette)
    # 1: Cyan/Blue, 2: Violet/Purple, 4: Pink/Magenta
    colors = {
        1: "#38bdf8",  # Sky 400
        2: "#a855f7",  # Purple 500
        4: "#f43f5e",  # Rose 500
    }

    # Create the figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0f172a")
    fig.suptitle("Gemma 4 Custom Benchmark Analysis", fontsize=18, fontweight="bold", color="#f8fafc", y=0.96)

    # --- SUBPLOT 1: Token Throughput ---
    ax1 = axes[0]
    ax1.set_title("Token Generation Rate (tokens/sec)", fontsize=13, fontweight="semibold", pad=15)
    ax1.grid(True)

    for c in concurrencies:
        sub_df = df[df["concurrency"] == c]
        ax1.plot(
            sub_df["max_tokens"],
            sub_df["tokens_sec"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=colors.get(c, "#ffffff"),
            label=f"Concurrency: {c}",
        )

    ax1.set_xlabel("Max Output Tokens", labelpad=10)
    ax1.set_ylabel("Tokens per Second", labelpad=10)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax1.set_xticks([4, 8, 16, 32, 64])

    # --- SUBPLOT 2: Latency (Avg & P95) ---
    ax2 = axes[1]
    ax2.set_title("Inference Latency (Avg vs P95)", fontsize=13, fontweight="semibold", pad=15)
    ax2.grid(True)

    for c in concurrencies:
        sub_df = df[df["concurrency"] == c]
        # Avg Latency: Solid line with circles
        ax2.plot(
            sub_df["max_tokens"],
            sub_df["avg_latency"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=colors.get(c, "#ffffff"),
            label=f"C={c} Avg",
        )
        # P95 Latency: Dashed line with x's
        ax2.plot(
            sub_df["max_tokens"],
            sub_df["p95_latency"],
            marker="x",
            markersize=6,
            linewidth=1.5,
            linestyle="--",
            color=colors.get(c, "#ffffff"),
            alpha=0.8,
            label=f"C={c} P95",
        )

    ax2.set_xlabel("Max Output Tokens", labelpad=10)
    ax2.set_ylabel("Latency (seconds)", labelpad=10)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax2.set_xticks([4, 8, 16, 32, 64])

    # --- SUBPLOT 3: Request Throughput ---
    ax3 = axes[2]
    ax3.set_title("Request Throughput (req/sec)", fontsize=13, fontweight="semibold", pad=15)
    ax3.grid(True)

    for c in concurrencies:
        sub_df = df[df["concurrency"] == c]
        ax3.plot(
            sub_df["max_tokens"],
            sub_df["throughput_req_sec"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=colors.get(c, "#ffffff"),
            label=f"Concurrency: {c}",
        )

    ax3.set_xlabel("Max Output Tokens", labelpad=10)
    ax3.set_ylabel("Requests per Second", labelpad=10)
    ax3.set_xscale("log", base=2)
    ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax3.set_xticks([4, 8, 16, 32, 64])

    # Single elegant legend for the whole figure
    # Let's put it on the first axis or clean it up individually
    ax1.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left")
    ax2.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper left", ncol=2, fontsize=9)
    ax3.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", loc="upper right")

    plt.tight_layout(rect=(0, 0.03, 1, 0.92))

    # Save the output image
    output_filename = "custom_benchmark_results.png"
    plt.savefig(output_filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Successfully generated and saved graphic to: {output_filename}")


if __name__ == "__main__":
    generate_plots()
