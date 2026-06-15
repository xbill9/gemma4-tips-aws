# 💰 Cost-Per-Token Analysis: RTX 6000 Pro GPU vs TPU v6e-4

This analysis evaluates the financial cost of serving **Gemma 4 (31B)** on Google Cloud Platform (GCP) using the benchmark results from both the **RTX 6000 Pro GPU (Cloud Run)** and the **Cloud TPU v6e-4**.

---

## 💵 GCP Hardware Cost Infrastructure

### 1. Cloud Run NVIDIA RTX 6000 Pro (Single Instance)
Cloud Run billing is calculated per second for allocated resources (Tier 1 region rates):
*   **vCPU:** $0.000024 per vCPU-second $\times$ 20 vCPUs = $0.00048 / s
*   **Memory (RAM):** $0.0000025 per GiB-second $\times$ 80 GiB = $0.00020 / s
*   **GPU (RTX 6000 Pro):** $0.00036522 / s
*   **Total Cloud Run Cost:** **$0.00104522 per second** (or **$3.76 per hour**) when active.

### 2. Cloud TPU v6e-4 (4x Trillium VM Slice)
Cloud TPU pricing is bundled (includes host VM CPU and RAM):
*   **TPU v6e Chip-hour Rate:** $1.375 per chip-hour
*   **Total TPU v6e-4 Cost:** $1.375 $\times$ 4 chips = **$5.50 per hour** (or **$0.00152778 per second**).

---

## 📈 Cost Per Million Tokens (At Concurrency = 16)

We evaluate cost efficiency under a standard concurrent workload (16 active users):

### Scenario A: Short Context (8 Input Tokens, 128 Generated Tokens)
*   **RTX 6000 Pro GPU:**
    *   Throughput: **299.53 tokens/second**
    *   Cost per Token: $0.00104522 / 299.53 = \$3.49 \times 10^{-6}$
    *   **Cost per Million Tokens:** **$3.49**
*   **Cloud TPU v6e-4:**
    *   Throughput: **306.10 tokens/second**
    *   Cost per Token: $0.00152778 / 306.10 = \$4.99 \times 10^{-6}$
    *   **Cost per Million Tokens:** **$4.99**
*   *Verdict:* **GPU is 30% cheaper** for small contexts under moderate load due to lower machine-hour pricing.

### Scenario B: Long Context (16,384 Input Tokens, 128 Generated Tokens)
*   **RTX 6000 Pro GPU:**
    *   Throughput: **246.62 tokens/second**
    *   Cost per Token: $0.00104522 / 246.62 = \$4.24 \times 10^{-6}$
    *   **Cost per Million Tokens:** **$4.24**
*   **Cloud TPU v6e-4:**
    *   Throughput: **2,892.37 tokens/second**
    *   Cost per Token: $0.00152778 / 2892.37 = \$5.28 \times 10^{-7}$
    *   **Cost per Million Tokens:** **$0.53**
*   *Verdict:* **TPU is 8x cheaper** ($0.53 vs $4.24 per million tokens) for long-context workloads because the TPU's memory bandwidth keeps throughput extremely high (2,892 tok/s).

---

## 📊 Scale-to-Zero and Utilization Adjustments

When factoring in overall utilization, Cloud Run's scale-to-zero capability shifts the metrics:

1.  **Low/Spiky Workloads (SRE/Ops Diagnostics):**
    *   If active usage is only **1 hour per day**, Cloud Run GPU costs **$3.76/day**.
    *   An on-demand TPU VM slice costs **$132.00/day** (since it does not scale to zero).
    *   *Recommendation:* **Use GPU (Cloud Run)**.
2.  **High/Constant Workloads (Production Chatbots/API Gateways):**
    *   If active usage is **24/7**, TPU VM slices run continuously at maximum efficiency.
    *   *Recommendation:* **Use TPU v6e-4**.
