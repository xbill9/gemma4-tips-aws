# 📊 Gemma 4 QAT L4 GPU 2D Grid Benchmark Summary

This report covers the performance metrics for the self-hosted **Gemma 4 26B QAT** model (`google/gemma-4-26B-A4B-it-qat-w4a16-ct`) deployed on a single **NVIDIA L4 GPU** (Cloud Run Gen2), running a sweep from **1 to 2048 concurrent users** across context sizes from **8 to 16,384 tokens**.

---

## 🕒 Average Latency Matrix (seconds)

| Context \ Users | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | 0.12s | 0.13s | 2.65s | 0.16s | 0.32s | 0.49s | 0.80s | 1.46s | 1.89s | 5.14s | 10.44s | 25.13s |
| **16** | 0.17s | 0.15s | 0.18s | 0.22s | 0.39s | 0.46s | 0.75s | 1.37s | 2.60s | 5.13s | 11.35s | 24.99s |
| **32** | 0.23s | 0.22s | 0.23s | 0.35s | 0.50s | 0.66s | 1.13s | 2.13s | 4.75s | 8.01s | 21.90s | 31.98s |
| **64** | 0.27s | 0.21s | 0.22s | 0.28s | 0.49s | 0.63s | 1.10s | 2.05s | 4.00s | 7.87s | 16.17s | 32.09s |
| **128** | 1.93s | 0.23s | 0.22s | 0.30s | 0.53s | 0.67s | 1.15s | 2.15s | 4.13s | 8.22s | 17.16s | 32.31s |
| **256** | 1.13s | 0.21s | 0.23s | 0.29s | 0.55s | 0.74s | 1.14s | 2.13s | 4.35s | 8.24s | 16.46s | 32.15s |
| **512** | 0.18s | 0.17s | 0.22s | 0.29s | 0.52s | 0.67s | 1.16s | 2.14s | 4.19s | 8.59s | 21.85s | 32.61s |
| **1024** | 2.12s | 0.22s | 0.24s | 0.29s | 0.47s | 0.71s | 1.22s | 2.62s | 4.85s | 8.98s | 17.09s | 31.65s |
| **2048** | 4.59s | 0.25s | 0.27s | 0.33s | 0.47s | 0.81s | 1.43s | 2.51s | 4.73s | 9.53s | 22.70s | 32.75s |
| **4096** | 9.52s | 0.26s | 0.32s | 0.39s | 0.59s | 0.87s | 1.63s | 3.22s | 6.54s | 11.85s | 24.24s | 32.67s |
| **8192** | 32.35s | 0.31s | 0.34s | 0.49s | 0.75s | 1.31s | 3.08s | 4.52s | 8.89s | 17.85s | 30.52s | 43.32s |
| **16384** | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s | 0.00s |

> [!NOTE]
> Latencies measured above `1024` concurrent users reflect request queuing delays in the client/server pipelines.

---

## 🚀 Throughput Matrix (Requests per second)

| Context \ Users | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **8** | 7.7 | 13.3 | 1.5 | 39.9 | 33.9 | 40.2 | 45.3 | 47.9 | 70.0 | 51.0 | 50.1 | 47.4 |
| **16** | 5.6 | 11.8 | 20.8 | 29.8 | 28.6 | 41.7 | 47.5 | 49.7 | 51.1 | 51.3 | 44.8 | 46.7 |
| **32** | 4.2 | 8.9 | 16.6 | 19.0 | 21.9 | 29.3 | 31.2 | 31.5 | 29.0 | 32.7 | 24.4 | 32.3 |
| **64** | 3.7 | 8.8 | 17.3 | 22.4 | 22.3 | 30.5 | 32.0 | 32.8 | 32.9 | 33.2 | 31.6 | 31.0 |
| **128** | 0.5 | 8.2 | 17.2 | 21.4 | 21.8 | 28.8 | 30.7 | 31.5 | 31.9 | 31.9 | 30.7 | 31.5 |
| **256** | 0.9 | 9.0 | 16.2 | 21.7 | 21.5 | 25.4 | 30.9 | 31.3 | 30.1 | 30.2 | 32.0 | 32.2 |
| **512** | 5.3 | 10.7 | 16.7 | 21.8 | 21.3 | 28.8 | 30.6 | 31.4 | 31.3 | 29.2 | 23.8 | 30.9 |
| **1024** | 0.5 | 8.5 | 15.8 | 21.0 | 23.2 | 26.9 | 28.8 | 24.1 | 27.9 | 29.3 | 29.6 | 29.8 |
| **2048** | 0.2 | 7.6 | 13.8 | 19.4 | 23.3 | 23.9 | 24.9 | 26.8 | 27.9 | 27.5 | 22.7 | 27.3 |
| **4096** | 0.1 | 7.2 | 12.0 | 16.0 | 17.9 | 21.4 | 21.4 | 20.6 | 20.0 | 21.9 | 21.2 | 19.8 |
| **8192** | 0.0 | 6.4 | 10.3 | 12.6 | 13.3 | 13.4 | 12.2 | 14.9 | 14.6 | 14.5 | 14.2 | 12.1 |
| **16384** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 📈 Concurrency Success Rates (At max load of 2048 users)

Despite massive parallel queues, the QAT model shows remarkable survival metrics thanks to the VRAM headroom freed up by QAT INT4 model weights:

*   **Up to 256 Tokens Context**: **100.0%** success rate at 2048 users.
*   **512 Tokens Context**: **96.2%** success rate at 2048 users.
*   **1024 Tokens Context**: **93.2%** success rate at 2048 users.
*   **2048 Tokens Context**: **85.7%** success rate at 2048 users.
*   **4096 Tokens Context**: **62.9%** success rate at 2048 users.
*   **8192 Tokens Context**: **37.9%** success rate at 2048 users.

---

## 💡 Key SRE Analysis & Explanations

### 1. The 16K (16,384) Context Limit Behavior
The 16K context window benchmark reported **0.0% success rate** across all concurrency configurations.
*   **Reason**: The Cloud Run vLLM container was deployed with `--max-model-len=16384`.
*   **Trigger**: In our test sweep, the input context is exactly `16384` tokens. However, the client requests a completion of `max_tokens=1` (the response token). This requires a context capacity of **16,385 tokens**, exceeding the engine's hard cap of 16384. vLLM rejected these requests with a prompt range validation error immediately.

### 2. High-Concurrency Latency Inflation
At **1024** and **2048** concurrent users, latencies for small prompts (8–128 tokens) scaled to **10s–32s**.
*   **Reason**: Cloud Run container concurrency was set to `--concurrency=4` and limited to `max-instances=1`. Thus, while the client fires 2048 requests in parallel, the Cloud Run instance only processes 4 request threads concurrently on the GPU, bottlenecking the rest in the request queues. This demonstrates that queuing delays—not GPU execution overhead—dominate at extreme concurrency limits.
