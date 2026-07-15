# Gemma-4 31B Inferentia port — findings (live-hardware run, 2026-07-15)

Ran the scaffold on a real inf2.24xlarge spot box (us-west-2d, 12 Neuron cores, 369 GB host).
Weights `google/gemma-4-31B-it` downloaded + verified (2 shards, 62.5 GB). Got as far as a
**successful single-rank compile** (`Compiler status PASS`), blocked on multi-rank orchestration.

## ✅ Confirmed on real weights

- **Dense**, `enable_moe_block=False`. 60 layers, hidden 5376, vocab 262144, softcap 30, tied embed.
- **Two attention layouts** (the crux — matches the "head size / KV layout" pain):
  - `sliding_attention` × 50: head_dim 256, 32 q-heads, **16 kv-heads**, has v_proj.
  - `full_attention` (global) × 10: head_dim **512**, 32 q-heads, **4 kv-heads**, **v_proj=None** (`attention_k_eq_v=True`, V reuses K).
- No PLE (`hidden_size_per_layer_input=0`), no kv-shared layers (all 60 non-shared).
- **TP=8 locked** (heads 32 / kv 16 divide 8; TP=12 fails).

## ✅ Solved

1. **KV-layout sharding** (`_shard`): shard q/k/v/o for the 50 sliding layers (16 kv → 2/rank);
   **replicate** attention for the 10 global layers (4 kv < 8 ranks, v_proj None) with full 4-head
   KV on every rank. MLP sharded everywhere. Diagnostic prints the distribution.
2. **Host memory**: two fixes — (a) load HF model bf16 not fp32; (b) `col()/row()` pass
   `dtype=o.weight.dtype` so parallel layers stay bf16 (default upcasts to fp32, doubling RAM).
3. **Compile memory**: `max_parallel_compilations=1` serializes per-rank build+compile →
   **peak 83 GB (was 300+)**, `Compiler status PASS`. The 300+ GB was the non-SPMD tracer
   compiling all 8 ranks in-process at once, NOT the model's real footprint (~15 GB/rank bf16).

## ❌ The remaining blocker — non-SPMD orchestration doesn't scale to 31B

- **Default (all 8 ranks compile at once): OOM** (~300 GB > 369 with overhead).
- **`max_parallel_compilations=1`: memory fine, but DEADLOCKS at multi-rank weight collection** —
  worker blocked on `pipe_read`, CPU idle, 25 min no progress. The mp_q/rendezvous collection
  hangs for a model this size.

## ⚠️ SPMD mode has its own obstacle here

`parallel_model_trace(spmd_mode=True, serialization_path=..., checkpoint_loader_callable=...)` is
the designed large-model path (compile 1 rank, serialize weights per-rank — no 8× compile, no
Queue deadlock). BUT `_load_weights` builds the model on **meta device** and injects weights from
the checkpoint via `get_sharded_checkpoint`. **The tp_alias recipe's device-aliased KV buffers are
runtime state, not checkpoint weights**, so they don't map onto the checkpoint-loading flow.
Forcing tp_alias → SPMD is uncertain.

## ➡️ Recommendation: port 31B via ModelBuilder (like the 12B), not tp_alias

The manual tp_alias recipe (from E4B) doesn't scale to 31B: its all-ranks-in-process compile OOMs,
and its aliased-KV design fights SPMD. The **12B port used NxD ModelBuilder**, which handles
single-rank/SPMD compile + per-rank weight loading + KV cache natively. That is the right base for
31B. Reuse the 12B `tp_mb.py` recipe with the 31B config + the global-layer replicate rule above.

## Infra learnings (for the next run)

- **Spot capacity for inf2.24xlarge is near-zero** (placement scores 1–3/10 across us-east-1/2,
  us-west-2). us-east-2 got reclaimed in ~20 min; us-west-2d held. inf2.48xlarge (768 GB) is
  **quota-blocked** (192 vCPU > 96 spot quota).
- **Drive the box via SSM** (no SSH key): use `aws-elasticbeanstalk-ec2-role` (has SSM core + S3).
  HF token via Secrets Manager (`hf_token`, us-east-1) with a scoped role policy.
- **After `pkill -9` of a Neuron process, REBOOT** — SIGKILL leaves cores in a bad state
  (`NRT_FAILURE` / `nrt_infodump` hang on next run). Verified 3×.
- Root disk fills; attach a separate EBS volume for weights (`/data`). fp32 host load of 31B is
  ~121 GB (fits 369), but bf16 is the right choice for tracing anyway.
