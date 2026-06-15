import asyncio
import csv
import os
import statistics
import time

import httpx
import matplotlib.pyplot as plt


def discover_vllm_url(service_name="gpu-12b-l4-devops-agent"):
    if os.getenv("VLLM_BASE_URL"):
        return os.getenv("VLLM_BASE_URL")
    cmd = [
        "gcloud",
        "run",
        "services",
        "describe",
        service_name,
        "--project=aisprint-491218",
        "--region=us-east4",
        "--format=value(status.url)",
    ]
    import subprocess

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception as e:
        print(f"Error discovering URL: {e}")
    return None


def get_auth_token():
    import subprocess

    try:
        return (
            subprocess.check_output(["gcloud", "auth", "print-identity-token"], stderr=subprocess.DEVNULL, timeout=10)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return ""


async def tokenize_prompt(url, token, prompt_text):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{url.rstrip('/')}/tokenize",
                json={"model": "google/gemma-4-12B-it", "prompt": prompt_text},
                headers=headers,
                timeout=15,
            )
            if res.status_code == 200:
                return res.json().get("count", len(prompt_text.split()))
        except Exception:
            pass
    return len(prompt_text.split())


async def get_prompt_for_size(url, token, size):
    word = " hello"
    guess_text = word * size
    count = await tokenize_prompt(url, token, guess_text)
    if count == size:
        return guess_text
    attempts = 0
    while count != size and attempts < 10:
        if count < size:
            guess_text += word * (size - count)
        else:
            guess_text = guess_text[: guess_text.rfind(word)]
        count = await tokenize_prompt(url, token, guess_text)
        attempts += 1
    return guess_text


async def main():
    url = discover_vllm_url()
    token = get_auth_token()

    print("Generating clean prompt for size 8...")
    prompt_text = await get_prompt_for_size(url, token, 8)

    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def send_req(client, sem):
        payload = {
            "model": "google/gemma-4-12B-it",
            "prompt": prompt_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start = time.perf_counter()
            try:
                res = await client.post(
                    f"{url.rstrip('/')}/v1/completions", json=payload, headers=headers, timeout=60.0
                )
                latency = time.perf_counter() - start
                if res.status_code == 200:
                    return {"success": True, "latency": latency}
                else:
                    return {"success": False, "error": f"HTTP {res.status_code}"}
            except Exception as e:
                latency = time.perf_counter() - start
                return {"success": False, "error": type(e).__name__, "latency": latency}

    new_results = []
    print("Re-running sweep for Context Size 8...")
    for c in concurrencies:
        sem = asyncio.Semaphore(c)
        limits = httpx.Limits(max_connections=c + 50, max_keepalive_connections=c + 50)

        start_time = time.perf_counter()
        async with httpx.AsyncClient(limits=limits, timeout=90.0) as client:
            tasks = [send_req(client, sem) for _ in range(c)]
            batch_results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

        successes = [r for r in batch_results if r["success"]]
        failures = [r for r in batch_results if not r["success"]]

        success_rate = len(successes) / c
        latencies = [r["latency"] for r in successes]

        if latencies:
            avg_lat = statistics.mean(latencies)
            p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0.0
        else:
            avg_lat, p95_lat = 0.0, 0.0

        req_per_sec = len(successes) / total_time if total_time > 0 else 0.0
        print(
            f"Concurrency {c:4d}: Success Rate: {success_rate:6.1%} | Avg Lat: {avg_lat:5.2f}s | Req/s: {req_per_sec:6.2f}"
        )

        new_results.append(
            {
                "context_size": 8,
                "concurrency": c,
                "success_rate": success_rate,
                "avg_latency": avg_lat,
                "p95_latency": p95_lat,
                "req_per_sec": req_per_sec,
                "total_time": total_time,
                "num_success": len(successes),
                "num_failure": len(failures),
            }
        )

    # Read existing CSV
    rows = []
    if os.path.exists("benchmark_sweep_results.csv"):
        with open("benchmark_sweep_results.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Keep rows that are NOT size 8
                if int(row["context_size"]) != 8:
                    rows.append(row)

    # Add new size 8 rows (convert values to strings matching format)
    for r in new_results:
        rows.append({k: str(v) for k, v in r.items()})

    # Sort rows by context_size (int) and concurrency (int)
    rows.sort(key=lambda x: (int(x["context_size"]), int(x["concurrency"])))

    # Save CSV
    with open("benchmark_sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("CSV updated.")

    # Regenerate Markdown
    context_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    latency_matrix: dict[int, dict[int, str]] = {size: {} for size in context_sizes}
    throughput_matrix: dict[int, dict[int, str]] = {size: {} for size in context_sizes}

    for r in rows:
        size = int(r["context_size"])
        c = int(r["concurrency"])
        latency_matrix[size][c] = f"{float(r['avg_latency']):.2f}s"
        throughput_matrix[size][c] = f"{float(r['req_per_sec']):.1f}"

    md_file = "benchmark_report.md"
    with open(md_file, "w") as f:
        f.write("# 📊 Gemma 4 vLLM GPU 2D Grid Concurrency Benchmark Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} (patched)\n")
        f.write(f"Endpoint: `{url}`\n")
        f.write("Model: `gemma-4-E4B-it` (12B on NVIDIA L4 GPU Cloud Run, 3 instances max)\n\n")

        f.write("## 🕒 Average Latency Matrix (seconds)\n\n")
        header = "| Context \\ Users | " + " | ".join(f"{c}" for c in concurrencies) + " |\n"
        separator = "|---:|" + "|".join("---:" for _ in concurrencies) + "|\n"
        f.write(header)
        f.write(separator)
        for size in context_sizes:
            row_str = f"| **{size}** | " + " | ".join(latency_matrix[size][c] for c in concurrencies) + " |\n"
            f.write(row_str)

        f.write("\n## 🚀 Throughput Matrix (Requests per second)\n\n")
        f.write(header)
        f.write(separator)
        for size in context_sizes:
            row_str = f"| **{size}** | " + " | ".join(throughput_matrix[size][c] for c in concurrencies) + " |\n"
            f.write(row_str)

    print("Markdown report updated.")

    # Regenerate Chart
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        plot_sizes = [8, 128, 1024, 8192, 16384]
        markers = ["o", "s", "d", "^", "v"]

        for size, marker in zip(plot_sizes, markers, strict=False):
            subset = [r for r in rows if int(r["context_size"]) == size]
            concs = [int(r["concurrency"]) for r in subset]
            avg_lats = [float(r["avg_latency"]) for r in subset]
            ax1.plot(concs, avg_lats, marker=marker, label=f"Context: {size} tokens")

        ax1.set_xscale("log")
        ax1.set_xticks(concurrencies)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.set_ylabel("Average Latency (seconds)")
        ax1.set_title("Gemma 4 Concurrency Sweep: Latency vs. Concurrent Users")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.legend()

        for size, marker in zip(plot_sizes, markers, strict=False):
            subset = [r for r in rows if int(r["context_size"]) == size]
            concs = [int(r["concurrency"]) for r in subset]
            reqs = [float(r["req_per_sec"]) for r in subset]
            ax2.plot(concs, reqs, marker=marker, label=f"Context: {size} tokens")

        ax2.set_xscale("log")
        ax2.set_xticks(concurrencies)
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.set_xlabel("Concurrent Users")
        ax2.set_ylabel("Throughput (Requests/sec)")
        ax2.set_title("Gemma 4 Concurrency Sweep: Throughput (Req/s) vs. Concurrent Users")
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("benchmark_chart.png", dpi=300)
        plt.close()
        print("Performance chart updated.")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    asyncio.run(main())
