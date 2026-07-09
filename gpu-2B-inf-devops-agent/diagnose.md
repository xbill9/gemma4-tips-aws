# Gemma 4 E2B on inf2 — Garbage/Empty Token Diagnosis

**Date:** 2026-07-03
**Symptom history:** model produced random characters; as of today it produces *empty* output — greedy decoding emits 200 consecutive invisible tokens (`content: ""`, `finish_reason: "length"`, empty logprobs) on both `/v1/chat/completions` and `/v1/completions`.

## What the live probes showed (2026-07-03)

- Server boots cleanly and serves (startup complete 13:10:48, port 8080).
- `"is a hotdog a sandwich"` → 200 completion tokens, empty content, ~41.6 tok/s wall.
- `logprobs` requests: chat returns `logprobs.content: []` (patch 32's guard silently skips
  everything because the Neuron on-device sampler returns no logprobs); completions 500s
  with "list index out of range" (unpatched path, same root cause).
- A constant invisible token under greedy = argmax always lands on the same special token
  (almost certainly token 0, `<pad>`, stripped by the detokenizer). That means the logits
  reaching the sampler are **degenerate — all-zero or NaN** — not merely noisy.
- Boot log's scary stack trace was NOT a crash: it was the leftover `traceback.print_stack()`
  debug in the trace.py sharding patch, firing on `layers.15/30.mlp.gate_proj` (the
  double-wide MLP layers — i.e. those layers were going through the patched fallback path).

## Root causes found in `apply_all_patches.py` (ranked)

### 1. trace.py auto-pad silently zero-filled mismatched weights  — top suspect for empty output
Patch 20c padded ANY checkpoint/module shape mismatch with zeros. A mismatched
`lm_head`/`embed_tokens`/MLP weight gets partially or fully zeroed → all-zero logits →
argmax token 0 → `<pad>` forever. **Fixed:** pad is now allowed only for the legitimate
hybrid-head-dim case (last-dim growth on `k_proj`/`v_proj`); everything else raises a
RuntimeError naming the tensor. Next boot will identify any corrupted mapping explicitly.

### 2. PLE weight injection could silently no-op
Patch 39g read PLE keys from a **single arbitrary safetensors shard**
(`glob(...)[0]`, unsorted, hardcoded `models--google--gemma-4-E2B-it` path) inside a
`try/except: print`. Multi-shard checkpoint ⇒ most PLE keys never injected ⇒
`per_layer_input_gate` / `per_layer_projection` stay at random init, and that noise is
added into hidden states at every one of the 35 layers. **Fixed:** reads all shards
(`models--google--gemma-4-E2B*` glob), logs injected key count, raises if config declares
`hidden_size_per_layer_input` but 0 keys were found.

### 3. KV-sharing boundaries contradicted the spec
Spec (`gemma4_2b_technical_specs.md`): virtual layers are `idx >= 20`; local layers
redirect to physical **18**, global to **19**. Patches used `>= 15` with sources **13/14**
(likely conflated with the double-wide-MLP boundary, which genuinely is 15). Result:
physical layers 15–19 ran with the wrong K/V on **every token** — sufficient on its own to
explain the original random-character output. **Fixed** in patch 2d
(`gemma4_shared_kv = layer_idx >= 20`, `gemma4_store_kv = layer_idx in (18, 19)`) and
patch 36 (`idx >= 20 → 18/19`). MLP boundary left at 15.

### 4. Attention logit softcap forced to 50.0
Five sites used `getattr(cfg, "attn_logit_softcapping", None) or ... or 50.0`, applying
`tanh(QK/50)*50` in every layer even though the spec lists only
`final_logit_softcapping = 30.0` (attention-level softcap is a Gemma-2 feature, dropped in
Gemma 3/3n-style archs). Distorts all attention. **Fixed:** plain getattr, skip when None.
(The lm_head softcap at 30.0 is kept — tanh is monotonic, can't cause the constant-token
symptom.)

### 5. Token-gen SWA mask assumed a linear cache; caches are ring buffers
The merged `patch_swa_mask` block masked with `idx >= pos - window + 1` over raw cache
indices, but writes wrap positions `% sliding_window`. Past position 512 it kept stale
entries and dropped the newest; past `2*window-1` (~1023 — exactly the 1,024 bucket) the
mask went **all-false → softmax over all -inf → NaN logits → argmax token 0**. A second
independent route to the empty-token symptom. The code's own comment said the wrapped
cache makes the mask unnecessary. **Fixed: removed entirely.**

### 6. `fill_prefix` kept the oldest window instead of the newest
`prefix_cache[..., :cache.shape[-2], :]` retained the *first* `window` tokens for SWA
layers; the update path uses `[-window:]`. Wrong window contents for prompts > 512 tokens
on the prefix-fill path. **Fixed:** `[..., -cache.shape[-2]:, :]`.

### 7. Debug noise removed
`SCALED_QK_DEBUG` stderr writes in `scaled_qk`, `DEBUG_SHARD` + `traceback.print_stack()`
in trace.py sharding (made every healthy boot look like a crash).

## Still open / needs verification

- **`softmax_scale=1.0`** (patch 2f2, replacing `sqrt(query_pre_attn_scalar)`): correct if
  Gemma 4 follows the Gemma3n convention (q/k-norm ⇒ scale 1.0); wrong (16× hot logits) if
  it kept Gemma-3 scaling. Verify with one CPU forward pass through the HF reference,
  comparing layer-0 attention scores.
- **`max_position_embeddings` set to 262144** (patch 2h) — that's the *vocab* size; spec
  context is 131,072. Numerically harmless below 128K but doubles the RoPE cache. Revert
  when convenient.
- **Global-attention layers get modulo-wrapped positions** (block manager `% max_slots`,
  `_get_index_to_update_new_position` else-branch): ring semantics on layers that must
  never wrap → silent overwrite of oldest global KV past cache capacity. Band-aid for the
  1006 OOB faults; real fix is sizing the global cache to the max context bucket. Related:
  natural `num_gpu_blocks=3` is overridden to 129 at boot.
- **Patch 32 logprobs guard** hides that the on-device sampler returns no logprobs (chat:
  silently empty; completions: still 500s). Decide whether to disable logprobs support or
  plumb them through.
- **RMSNorm +1.0 offsets removed globally** (patch 2f4) on the claim Gemma4 stores full
  weights (input_layernorm mean ~10.7 observed). If any norm tensor is actually
  zero-centered (mean ≈ 0), it now needs its +1.0 back — check per-tensor stats if quality
  is off after decoding works.
- **Server is exposed to the public internet with no auth** — scanners observed probing
  `/login`, `/hudson`. Lock SG port 8080 to trusted IPs or use SSM port-forwarding.

### 8. `use_qk_norm=True` broke tracing (found by redeploy #1, 2026-07-03 18:19 UTC)
First redeploy with the fixes died at engine init:
`ValueError: Unexpected new/changed parameters after trace: layers.N.self_attn.qk_norm.weight
before: None → after: [1024] (SWA) / [2048] (global)`.
Patch 2f3 had flipped `use_qk_norm` False→True to get q/k norms applied, but the base
class's `qk_norm` is a **fused** norm over flattened heads (4×256=1024 / 4×512=2048),
created **lazily on first forward** — after tracing starts — and it's the wrong op for
Gemma anyway (per-head norm over head_dim). Earlier boots only survived because they
loaded stale compiled artifacts ("init engine took 0.00 seconds"); the patch changes
forced a real re-trace and surfaced it. Side note: the 1024/2048 split confirms the
per-layer hybrid head_dim config is plumbing through correctly.
**Fixed:** `use_qk_norm` stays False; vendor `q_layernorm`/`k_layernorm` (weights already
in the checkpoint) are now applied explicitly pre-RoPE at the top of the
`apply_rotary_embedding` patch. Redeploy #2: SSM command
`94908b8a-e05f-4e6e-ae25-adf6c66b1d95`, ~16:38 local — full recompile expected.

## Deploy state

- Fixes applied to `apply_all_patches.py` (local), `py_compile` clean.
- Redeploy sent 2026-07-03 ~14:18 UTC-local via `run_deploy_on_existing_optimized.py`,
  SSM command `1c40c802-a667-4277-8b9b-38e7cf6377ba`, instance `i-0769ecb1ab94dc9a8`
  (us-east-2). Container recreated from clean image; traced code changed so expect a
  Neuron recompile (engine-ready timeout 1800s). `wait_and_verify.py` monitoring.
- **Expected outcomes:** either the model decodes real text, or one of the newly-fatal
  checks (auto-pad RuntimeError naming a tensor / PLE zero-key RuntimeError) fires at boot
  and pinpoints the remaining weight-mapping bug.

## Session 2 (2026-07-03 evening): redeploy #2 booted clean; empty output persists — sampler-input logits are exactly flat

Redeploy #2 (qk_norm fix) came up healthy at 20:46 UTC: full recompile passed, PLE
injection loaded **143 keys** (108 per_layer/embed keys + **35 `layers.N.layer_scalar`**),
the newly-fatal auto-pad check did NOT fire (so no weight got zero-padded), and the
server serves at ~41 tok/s. Output is still 100% invisible tokens.

### Probes and what they proved

- Greedy + `skip_special_tokens:false` → `<pad>` (token 0) every step, both endpoints.
- temp=1.0 (default top_k=64 from generation_config.json) → only `<unusedNN>`/`<unk>`/`<pad>`,
  all token ids < 64, near-uniform.
- temp=1.0 + top_k=200 → ids spread up to ~200 (`<unused70>`, markup tokens, `<tool_call|>`).
- temp=1.0 + top_k=-1 (disabled) → STILL only low-id special tokens (sampler's global
  top-k cap ~256). If logits weren't tied, disabling top_k would sample real words from
  the 262k vocab. It didn't.
- **Conclusion: the logits the sampler sees are exactly tied across the vocab** —
  all-zero, NaN, or an unwritten output buffer. Top-k of tied logits returns the first k
  vocab indices; argmax returns index 0 = `<pad>`. This explains every symptom.

### Suspects cleared this session

- **RMSNorm +1.0 removal (2f4): CORRECT.** Checkpoint stats: every language norm is
  stored full — `model.language_model.norm.weight` mean 14.17 (min ≈ 0),
  input_layernorm mean 10.58, q_norm ≈ 1.0, k_norm ≈ 0.11 (genuinely small scale).
  Closes the open question; nothing needs +1.0 back.
- **lm_head tie: WORKING.** Checkpoint has no `lm_head.weight` (tied);
  `tie_word_embeddings=True` verifiably propagates into the built
  `Gemma3InferenceConfig`, so `update_state_dict_for_tied_weights` fires
  (application_base.py:731 → modeling_gemma3.py:671).
- **layer_scalar trunk-multiplication (39f): CORRECT placement.** Checkpoint carries 35
  scalars (0.018–0.87). Official HF `modeling_gemma4.py` does exactly
  `hidden_states *= self.layer_scalar` at layer end (GGUF calls it
  `LAYER_OUT_SCALE`). Numerically fine: each layer's branches read RMSNorm-ed input, so
  deltas re-enter at O(1); the trunk does not vanish.

### Current experiment (in flight): CPU sampling to expose real logits

The on-device Neuron sampler is the last black box (it also returns no logprobs — the
thing patch 32 papers over). Redeployed ~22:50 UTC (SSM `9369d964-10e4-4abe-8b46-c75e188adb8d`)
with host-side sampling:

- `run_deploy_on_existing_optimized.py` edited (REVERT LATER for perf):
  `-e NEURON_ON_DEVICE_SAMPLING_DISABLED=1` +
  `--additional-config '{"override_neuron_config": {"on_device_sampling_config": null}}'`
  (validated: vllm_neuron 0.5.1 at `/opt/vllm/vllm_neuron` passes unknown override keys
  through; `cpu_sampler` is unconditionally initialized; fresh container ⇒ no
  pre-compiled-artifacts short-circuit; lm_head flips to gather_output=True ⇒ recompile).
- Expected outcomes: **(a)** real text + working logprobs ⇒ the model graph was fine all
  along and the on-device sampler (or its logits plumbing) is the bug;
  **(b)** still `<pad>` + logprobs show tied/zero/NaN values ⇒ the model graph emits
  degenerate logits — bisect upstream (final norm output, softcap hook, lm_head matmul).

### RESULT: outcome (b) — the compiled graph emits perfectly uniform logits

Two deploy iterations were needed:
1. First attempt crashed at engine init: `TypeError: 'module' object is not callable`
   at `neuronx_distributed_model_loader.py:81 self.sampler = Sampler()` — the plugin's
   `NEURON_ON_DEVICE_SAMPLING_DISABLED=1` env path is broken in vllm_neuron 0.5.1
   (imports `Sampler` as a module). **That env var is unusable; don't set it.** The
   `--additional-config '{"override_neuron_config": {"on_device_sampling_config": null}}'`
   override alone flips the runner to its always-initialized `cpu_sampler`.
2. Second deploy (override only) booted clean (~00:57 UTC 07-04) and logprobs now work.

With CPU sampling, greedy still emits `<pad>` and **every top-5 logprob is exactly
-12.476649 = -ln(262144) = -ln(vocab_size)**, top tokens = vocab ids 0-4 in order.
The distribution is EXACTLY uniform over the whole vocab ⇒ the traced Neuron graph
outputs a constant logit for all 262,144 tokens. On-device sampler exonerated (it was
faithfully reporting ties). The `patch 32` logprobs guard and the completions-endpoint
500 are collateral of on-device sampling, not separate bugs.

### Host-side state dict verified CLEAN (post-convert, pre-shard)

`NeuronGemma3ForCausalLM.get_state_dict()` run offline in the container:
`lm_head.weight` (262144, 1536) absmean 0.023964 — exactly tied to embed_tokens ✓;
`norm.weight` absmean 14.17 ✓; `rank_util.rank` = [0, 1] ✓; `layers.0.layer_scalar`
0.0178 ✓ (loads); fused `qkv_proj.q_proj` healthy ✓; `embed_tokens_per_layer` healthy ✓.

**Conclusion: every weight handed to sharding/tracing is correct. The zero/constant
is produced by the forward computation inside the compiled graph.** Since RMSNorm,
attention, MLP and the PLE gate are all zero-preserving (f(0)=0 with residual 0), a
single point that zeroes the stream early (prime suspect: the vocab-parallel
`ParallelEmbedding` gather under TP=2 masking every token out of range → exact-zero
embeddings) reproduces the end-to-end symptom exactly.

### CPU-mode ground truth (2026-07-04 ~01:28 UTC): the model computes NON-uniform logits on CPU

Ran the identical patched stack (same container, same patched modeling files, same
sharded-weight conversion) on CPU via the vendor-supported path:

```
docker exec -e NXD_CPU_MODE=1 vllm-server torchrun --nproc_per_node=2 \
  -m neuronx_distributed_inference.inference_demo \
  --model-type gemma3 --task-type causal-lm run \
  --model-path <snap> --compiled-model-path /tmp/cpu_check \
  --prompt 'The capital of France is' --torch-dtype float32 --tp-degree 2 \
  --batch-size 1 --seq-len 128 --max-context-length 128 --max-new-tokens 4 \
  --check-accuracy-mode skip-accuracy-check --on-cpu --skip-compile
```
(Needs torchrun for tp=2; float32 — gloo lacks bf16; loads in ~70 s; 123 GB host RAM.)

Output: `The capital of France is B)="<どちら 何เあるマ"); 以YL が...` — **garbage but
VARIED tokens**, i.e. real, non-degenerate logits. Two-layer diagnosis:

1. **The exact-uniform logits are serving/device-path specific** — the same math on CPU
   does not produce them. Prime suspect: `--async-scheduling` in the vLLM launch
   (async output sync reading the logits/token buffer before the device writes it →
   zeros → uniform logprobs + `<pad>` greedy). inference_demo (synchronous) shows real
   compute. Discriminating redeploy WITHOUT `--async-scheduling` in flight
   (SSM `d31ed8b8-b7d3-4ea0-b8ca-97b6b862d4bf`).
2. **Under that, the port still has a quality bug** — garbage text matches the original
   "random characters" era. Once the uniform-logits layer is fixed, debug quality with
   the tensor-capture equivalence flow
   (`neuron_agentic_development/artifacts/skills/neuron-framework-equivalence/references/dump-tensors.md`):
   hook embed/layers/norm/lm_head on CPU-mode vs HF reference (no gemma4 in container
   transformers 4.57.6 — reference must come from upstream transformers main).

### Next session: bisect inside the graph (no recompile needed for step 1)

1. **Eager CPU forward** in the container: instantiate the patched model classes (plain
   torch modules pre-trace), load the converted state dict, run `embed_tokens` +
   layer 0..N on a token id, print per-stage `abs().mean()` — find the first exact zero.
   Focus: ParallelEmbedding output under tp_degree=2 (also embed scale
   `* sqrt(hidden_size)` — check whether vendor base forward applies it or the patched
   PLE-path comment "vendor returns raw embeddings" indicates it's missing on the main
   path too).
2. If embeddings are nonzero in eager mode, dump graph I/O on device
   (`/tmp/neuron-input-dump` exists but is empty — find the flag that populates it) or
   binary-search with single-layer traces.
3. Deploy config note: `run_deploy_on_existing_optimized.py` currently carries the
   CPU-sampling `--additional-config` override for debugging (keep until decoding works,
   then remove to restore on-device sampling perf ~41 tok/s).

## Architecture cheat sheet (from `gemma4_2b_technical_specs.md`)

| Item | Value |
| --- | --- |
| Layers | 35 (KV: virtual ≥ 20 → sources 18 sliding / 19 full) |
| Attention cycle | period 5: layers with `(i+1) % 5 == 0` are global, rest SWA |
| head_dim | 256 local / 512 global (partial_rotary_factor 0.25, theta 1e6) |
| SWA window | 512 tokens (theta 10k) |
| MLP | 6144, double-wide 12288 on layers 15–34 |
| Heads | 8 attention / 1 KV |
| Softcap | final logits only, 30.0 |
| Context / vocab | 131,072 / 262,144 (don't swap these) |
