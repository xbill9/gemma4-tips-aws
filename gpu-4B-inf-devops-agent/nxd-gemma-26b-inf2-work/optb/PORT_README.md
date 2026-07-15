# Gemma-4 26B-A4B → Inferentia2 (inf2.24xlarge, TP=8) — port scaffold

HF: `google/gemma-4-26B-A4B-it`. **A4B = 26B total params, ~3.2B active/token (MoE).**
Seeded from the E4B `tp_alias` recipe, but the MoE block is **new capability** on Neuron —
this is a research spike, NOT a port. Config saved as `config.reference.json`.
**Recommendation on record: land 31B (dense) first, then tackle this.**

## Confirmed architecture (from config.json)

| field | value | implication |
|---|---|---|
| `enable_moe_block` | **True** | MoE — dense-MLP shard path is WRONG here (see gap below) |
| `num_experts` | **128** | all 128 resident in HBM |
| `top_k_experts` | **8** | 8 experts routed per token |
| `moe_intermediate_size` | 704 | per-expert FFN (tiny) |
| `intermediate_size` | 2112 | dense/shared FFN size (confirm if a shared expert exists) |
| `num_hidden_layers` | 30 | |
| `hidden_size` | 2816 | |
| `num_attention_heads` / `kv` | 16 / 8 | TP=8 → 2 heads, 1 kv/rank ✓ |
| `head_dim` | 256 | |
| `layer_types` | 25 sliding + 5 full | sliding_window=1024 |
| `hidden_size_per_layer_input` | **0** | NO PLE (like 31B) — "closer to 4B" is about *active* size, not PLE |
| `final_logit_softcapping` | 30.0 | keep ✓ |
| `tie_word_embeddings` | True | |
| `vocab_size` | 262144 | |
| `vision_config` | present | text path only |

## ⚠️ Memory: A4B saves COMPUTE, not MEMORY

- Total ~24.6B params → **bf16 RESIDENT weights ~49 GB**. **ALL 128 experts must sit in HBM**;
  top-8 routing only reduces FLOPs, not footprint. Budget for the full 26B, not 4B.
- At TP=8: ~6.1 GB/core weights (+ unsharded lm_head 2.82 GB/core). Fits 16 GB/core.
- Expert-parallel option: 128 experts / TP=8 = **16 experts/rank**.

## ⚠️ The MoE gap (the crux — new capability)

`tp_alias_trace.py` `_shard()` shards a **dense** MLP (`gate/up/down`). For 26B you must replace it
with an expert-aware path:

1. **Router** (`gate`, 2816→128): replicate (tiny), compute top-8 + softmax weights per token.
2. **128 experts**, each `gate/up/down` at `moe_intermediate_size=704`: shard across TP
   (expert-parallel: 16 experts/rank) or tensor-parallel within expert.
3. **Top-8 gather/scatter**: token→expert dispatch does NOT trace as a clean static graph on Neuron.
   Start with **dense/all-experts compute** (run all 128 experts on every token, mask by router
   weight — wasteful but static-shape and traceable) for first-light correctness, optimize later.
4. Confirm whether every layer is MoE or interleaved with dense (`intermediate_size=2112` suggests a
   possible shared/dense component — inspect the loaded module structure).

Neither NxD nor the optb stack has traced MoE before. `top_k_experts=8` gather/scatter is the risk.

## Bring-up sequence

1. Inspect the loaded `Gemma4` module to see the exact MoE block structure (router + experts + any
   shared FFN); decide expert-compute strategy (start all-experts-dense).
2. Download weights → `/workspace/real-gemma4-26B-A4B-it`; set `MP` (currently `real-gemma4-26B-it`).
3. Rewrite `_shard()` + `DecWrap.forward` for MoE; strip PLE (as in 31B).
4. `KV_MAX=256 KV_BUCKET=64 TP_DEGREE=8 python tp_alias_trace.py` → validate SEQ_MATCH.
5. Server + Dockerfile + publish.

Note: `MP` in the scaffold reads `/workspace/real-gemma4-26B-it` — update to `-26B-A4B-it`.
