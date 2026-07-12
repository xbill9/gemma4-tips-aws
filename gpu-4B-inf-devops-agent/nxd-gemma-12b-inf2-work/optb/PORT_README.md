# 12B port — TP=2 on inf2.8xlarge (device-prefill ModelBuilder stack)

Clean starting point copied from the working **E4B** device-prefill build
(`../nxd-gemma4-inf2-work/optb/`). Same architecture the E4B port landed on:
one weight-sharing `ModelBuilder` trace (prefill + decode buckets share **one** bf16
sharded weight set + on-device aliased KV), device prefill (~0.16 s), served by
`optb_server_mb.py`.

Target: **TP=2 on a single `inf2.8xlarge`** (both cores, 2×16 GB HBM).

## Files (verbatim from E4B, `MODEL_DIR`/`TP_DEGREE` now env-parameterized)
- `tp_mb.py` — build + save the device-prefill model (`MB_WDTYPE=bf16 MB_SAVE=... python tp_mb.py`)
- `optb_server_mb.py` — device-prefill HTTP server (OpenAI routes + `/generate`)
- `Dockerfile.mb` — package the built model + server

## What carries over unchanged (all the hard-won E4B fixes)
- **`layer_scalar` buffer load** — ModelBuilder shards *parameters* only, never buffers;
  `build_module()` copies the per-layer `layer_scalar` from the checkpoint by hand. Required
  again here (config-built module), already in `tp_mb.py`.
- On-device **aliased KV** (device-resident across prefill→decode), one-hot scatter KV write.
- **bf16 sharded weights** (`MB_WDTYPE=bf16`) — correct + fast; halves on-device weights.
- `torch.jit.save`/`load` of the executor + `initialize_with_saved_weights(start_rank)`.

## OPEN QUESTIONS — need the target checkpoint to resolve
1. **Which 12B, and is it Gemma-3n-style or a plain transformer?**
   `tp_mb.py`'s forward passes `inputs_embeds` **and** `per_layer_inputs` (`lang.get_per_layer_inputs`)
   — that Per-Layer-Embeddings (PLE) machinery is **3n-specific**. A standard Gemma-3 12B has **no PLE**,
   so the embed/forward path (`_inputs`, `embR`, dummy `embed_tokens`) must be simplified. This is the
   biggest branch point.
2. **Fit at TP=2.** 12B bf16 ≈ 24 GB → ~12 GB/rank across 2 cores, leaving ~4 GB/rank for KV +
   activations. Plausible but tight; may cap the context window (`KV_MAX`) smaller than E4B's 512,
   or need int4/QAT weights to breathe. Verify with a small-context trace first.
3. **GQA sharding.** Confirm `num_key_value_heads` vs `num_attention_heads` and `head_dim` for the 12B
   config. With `nkv % TP == 0` the existing 1-head-per-rank sharding (`_kv_rank_width`) is correct;
   otherwise KV replicates and `num_key_value_groups` needs re-deriving (already handled for the
   nkv%TP!=0 case, but re-verify the head geometry).

## Next step
Point `MODEL_DIR` at the staged 12B checkpoint, then:
`KV_MAX=256 KV_BUCKET=64 MB_WDTYPE=bf16 python tp_mb.py`  → expect `SEQ_MATCH True`, then scale context.
