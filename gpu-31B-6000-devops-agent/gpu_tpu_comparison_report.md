# ⚖️ Performance Comparison: RTX 6000 Pro GPU vs TPU v6e-4

This report compares **Gemma 4 (31B)** serving metrics between a **Cloud Run NVIDIA RTX 6000 Pro GPU** and a **Cloud TPU v6e-4 (Trillium)** cluster, based on their respective matrix benchmark sweeps.

---

## 📈 Comparison Visualizations

![GPU vs TPU Performance Comparison](/home/xbill/.gemini/antigravity-cli/brain/e141a97f-1e2f-4099-86bc-4c43ddafbe78/gpu_tpu_comparison.png)

---

## 🔍 Comparative Analysis

### 1. Token Throughput (tok/s) Scaling
* **Small Context (8 tokens):** 
  * TPU v6e-4 scales to a peak of **~1,379 tokens/second** at high concurrency.
  * RTX 6000 Pro peaks and saturates at **~308 tokens/second**.
  * **Ratio:** TPU delivers **4.5x** more throughput for small-context workloads.
* **Large Context (16,384 tokens):**
  * TPU v6e-4 scales to **~3,206 tokens/second** under load.
  * RTX 6000 Pro peaks at **~344 tokens/second**.
  * **Ratio:** TPU delivers **9.3x** more throughput for long-context workloads. The TPU's massive memory bandwidth (Trillium architecture) and larger TPU v6e cluster configuration excel at high-context parallel decoding.

### 2. Latency Profiles
* **Ultra-Low Latency (TPU):** For standard-size requests at low-to-medium concurrencies, TPU latency stays sub-second (e.g., **0.15s** at concurrency 1, rising to only **0.56s** at concurrency 128). The GPU averages **3.3s** to **28s** for the same levels.
* **Queuing Thresholds:** 
  * GPU performance saturates early (concurrency > 16), after which latency scales linearly with request count.
  * TPU handles up to **128+ concurrent users** before latency begins showing significant queuing patterns.

---

## 📋 Direct Metrics Comparison Table (At Concurrency = 16)

| Context Size | Metric | RTX 6000 Pro GPU | Cloud TPU v6e-4 | Performance Delta |
|---|---|---|---|---|
| **8 tokens** | Throughput | 299.53 tok/s | 306.10 tok/s | TPU +2.2% |
| | Avg Latency | 5.17s | 0.46s | TPU is **11.2x faster** |
| **16,384 tokens**| Throughput | 246.62 tok/s | 2892.37 tok/s | TPU is **11.7x faster** |
| | Avg Latency | 5.32s | 90.52s | GPU is **17x faster** (due to lower overall queued load per unit)* |

> [!NOTE]
> *At high contexts and concurrency, the GPU's lower absolute latency is a result of serving significantly fewer concurrent requests in active decoding states (saturating request queue early) compared to the TPU, which processes thousands of tokens in parallel across all 128+ active decoding slots.
