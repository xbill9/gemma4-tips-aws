import asyncio
import os
import statistics

# Import helpers from server.py
import sys
import time

import httpx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from server import (
    get_active_model_name,
    get_auth_token,
    get_vllm_client,
    get_vllm_url,
)


async def send_request(
    client: httpx.AsyncClient, base_url: str, model: str, prompt: str, max_tokens: int, headers: dict
) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    start_time = time.perf_counter()
    try:
        response = await client.post(base_url, json=payload, headers=headers, timeout=180)
        end_time = time.perf_counter()
        if response.status_code == 200:
            latency = end_time - start_time
            data = response.json()
            tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
            return {"success": True, "latency": latency, "tokens": tokens}
        else:
            return {"success": False, "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def run_sweep():
    url = get_vllm_url()
    token = get_auth_token()

    client = await get_vllm_client()
    model = await get_active_model_name(client)

    print(f"🚀 Starting GPU Matrix Benchmark Sweep against {url}")
    print(f"Model: {model}")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"{url.rstrip('/')}/v1/completions"

    # Concurrency and Context Window sizes
    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    context_sizes = [8, 64, 512, 4096, 16384]

    results = []

    # Warmup
    print("Warming up model...")
    async with httpx.AsyncClient() as http_client:
        await send_request(http_client, base_url, model, "Warmup", 16, headers)

    for ctx_size in context_sizes:
        # Construct a prompt of approximately ctx_size tokens (assuming ~4 chars per token)
        # We repeat the word " SRE" to reach the target length
        prompt = "SRE " * max(1, ctx_size - 128)  # subtract output tokens to fit window

        print(f"\n--- Context Window size: {ctx_size} tokens ---")

        for concurrency in concurrencies:
            # We want enough prompts to represent concurrency
            # For high concurrency, we run 'concurrency' requests
            # For low concurrency, we run a minimum of 5 requests
            num_prompts = max(concurrency, 5)

            # To avoid overloading or taking too long, cap total prompts at 128
            # (In a real benchmark sweep we can cap at 128 or 256 depending on load)
            if num_prompts > 128:
                num_prompts = 128

            print(f"Running Concurrency={concurrency} with {num_prompts} requests...")

            sem = asyncio.Semaphore(concurrency)
            async with httpx.AsyncClient() as http_client:
                start_batch = time.perf_counter()

                async def sem_request(sem=sem, prompt=prompt):
                    async with sem:
                        return await send_request(http_client, base_url, model, prompt, 128, headers)

                tasks = [sem_request() for _ in range(num_prompts)]
                batch_results = await asyncio.gather(*tasks)
                total_time = time.perf_counter() - start_batch

            successes = [r for r in batch_results if r["success"]]
            latencies = [r["latency"] for r in successes]

            if not latencies:
                print(f"❌ Concurrency={concurrency}, Context={ctx_size} failed entirely.")
                results.append(
                    {
                        "concurrency": concurrency,
                        "context_size": ctx_size,
                        "success_rate": 0.0,
                        "avg_latency": 0.0,
                        "p95_latency": 0.0,
                        "throughput_req_sec": 0.0,
                        "tokens_per_sec": 0.0,
                    }
                )
                continue

            avg_lat = statistics.mean(latencies)
            p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else avg_lat
            throughput = len(successes) / total_time
            tokens_per_sec = sum(r["tokens"] for r in successes) / total_time

            success_rate = len(successes) / num_prompts
            print(
                f"✅ Success Rate: {success_rate:.1%}, Throughput: {throughput:.2f} req/s, Tokens: {tokens_per_sec:.2f} tok/s, Avg Latency: {avg_lat:.2f}s"
            )

            results.append(
                {
                    "concurrency": concurrency,
                    "context_size": ctx_size,
                    "success_rate": success_rate,
                    "avg_latency": avg_lat,
                    "p95_latency": p95_lat,
                    "throughput_req_sec": throughput,
                    "tokens_per_sec": tokens_per_sec,
                }
            )

    # Save to CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matrix_benchmark_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n📊 Results saved to {output_file}")

    # Plot results
    plot_results(df)


def plot_results(df):
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

    # Unique context sizes
    context_sizes = sorted(df["context_size"].unique())
    colors = ["#38bdf8", "#a855f7", "#f43f5e", "#10b981", "#f59e0b"]
    color_map = {size: colors[i % len(colors)] for i, size in enumerate(context_sizes)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f172a")
    fig.suptitle("Gemma 4 RTX 6000 GPU Performance Sweep", fontsize=18, fontweight="bold", color="#f8fafc", y=0.96)

    # Subplot 1: Throughput (Tokens/s) vs Concurrency
    ax1 = axes[0]
    ax1.set_title("Token Generation Rate (tokens/sec) vs Concurrency", fontsize=13, fontweight="semibold", pad=15)
    ax1.grid(True)
    for size in context_sizes:
        sub_df = df[df["context_size"] == size].sort_values("concurrency")
        ax1.plot(
            sub_df["concurrency"],
            sub_df["tokens_per_sec"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=color_map[size],
            label=f"Context: {size} tokens",
        )
    ax1.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
    ax1.set_ylabel("Tokens per Second", labelpad=10)
    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax1.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    # Subplot 2: Latency vs Concurrency
    ax2 = axes[1]
    ax2.set_title("Average Latency (seconds) vs Concurrency", fontsize=13, fontweight="semibold", pad=15)
    ax2.grid(True)
    for size in context_sizes:
        sub_df = df[df["context_size"] == size].sort_values("concurrency")
        ax2.plot(
            sub_df["concurrency"],
            sub_df["avg_latency"],
            marker="o",
            markersize=6,
            linewidth=2.5,
            color=color_map[size],
            label=f"Context: {size} tokens",
        )
    ax2.set_xlabel("Concurrency (Concurrent Users)", labelpad=10)
    ax2.set_ylabel("Average Latency (s)", labelpad=10)
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax2.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155")

    plt.tight_layout(rect=(0, 0.03, 1, 0.92))

    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtx_6000_benchmark.png")
    plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Chart saved to {chart_path}")


if __name__ == "__main__":
    asyncio.run(run_sweep())
