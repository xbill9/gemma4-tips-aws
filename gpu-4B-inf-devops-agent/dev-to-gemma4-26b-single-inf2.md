---
title: "Squeezing a 26B MoE onto the *cheapest* Inferentia box — from a $6.49/hr 24xlarge to a $0.76/hr inf2.xlarge"
published: false
description: "The 26B-A4B MoE ran fine on an inf2.24xlarge. Getting it onto a single 16GB-host inf2.xlarge meant killing the fp4 fantasy, an all-int8 squeeze, a slim host-side deploy — and one embedding scale that turned Paris into a wall of spaces."
tags: machinelearning, aws, ai, python
# canonical_url: https://your-blog.example/gemma4-26b-single-inf2
# cover_image: https://.../cover.png                            # 1000x420 recommended
# series: "Gemma-4 on Inferentia"
---

> In the last entry I got Gemma-4's 128-expert MoE running on an `inf2.24xlarge` and signed off with a
> cliffhanger: fitting it on a 2-core box "needs fp4 — a separate expedition." This is that expedition.
> It ended nowhere near where I thought it would: **not** with fp4, and **not** on the 8xlarge I was
> aiming for, but on the smallest, cheapest Inferentia2 instance AWS sells — a **single `inf2.xlarge`
> with 16 GB of host RAM** — running a **26B-parameter model**. Here's the refinement trail, dead ends
> included.

The finished port (see the previous article) served the 26B-A4B on a 24xlarge: 12 NeuronCores, 192 GB
HBM, ~$6.49/hr on-demand. It worked, it was correct, and it was **overkill for one user asking one
question at a time.** The whole point of a "4B-active" MoE is that it's cheap to *run*; it shouldn't need
a datacenter-class box to *host*. The goal: get it onto the bottom-tier `inf2.xlarge` — 2 cores, 32 GB
HBM, and only **16 GB of host RAM** — at ~$0.76/hr. That's **8.6× cheaper**.

| | 24xlarge (shipped) | inf2.xlarge (goal) |
|---|---|---|
| NeuronCores | 12 (used 8) | **2** |
| HBM | 192 GB | **32 GB** |
| Host RAM | 768 GB | **16 GB** |
| On-demand | ~$6.49/hr | **~$0.76/hr** |

Two walls stood in the way: the model doesn't fit 32 GB of HBM at bf16, and it doesn't fit 16 GB of host
RAM the *normal* way. I had to break both.

## Wall 1: the memory math, and why "A4B" doesn't save you

The experts are ~93% of the weight. At bf16 the 128 experts are ~45.6 GB — replicate that across TP=2
and you're at ~22.8 GB **per core** against a 16 GB budget. Top-8 routing reduces *compute*, not
*footprint*: all 128 experts must be resident. So the experts alone blow the box.

My first move was the obvious one — **int8 the experts**. NxD has `QuantizedColumnParallel` /
`QuantizedRowParallel`, and (from the prior article) int8-per-channel on this model is *numerically
perfect*: token-for-token identical to fp32. Experts drop to ~11.4 GB/rank. Should fit.

It didn't. `Could not load the model status=4 Allocation Failure` — **~3–4 GB over** the 16 GB core. And
`inf2` has no 4-core SKU to split the difference. My conclusion at the time, which I even wrote down: to
close that last 3–4 GB you need **fp4** experts.

## The fp4 rabbit hole (two of them, both dead ends)

I chased fp4 twice.

**First**, `QuantizedColumnParallel` advertises `F4E2M1FN_X4`. But its state_dict weight is a **packed
`uint16`** microscaling format (32 fp4 → 8 uint16) produced by `from_float`/`preshard_hook` — nothing
like int8's five-line quantizer. Then I read the AWS docs plainly: **NxD Inference quantization supports
INT8 and FP8 only.** FP8 is 8-bit — same size as int8, so it doesn't help. `QuantizedDtype.F4E2M1FN_X4`
*exists* as a constant but isn't a wired-up inference path.

**Second**, on the newest Neuron 2.31 stack (neuronx-cc 2.26, NxD 0.19) fp4 looked *less* absent —
there's `BLOCKWISE_SYMMETRIC`, `block_size`, `from_float`. So I tried again, and hit the floor twice:

- The clean `QuantizedColumnParallel` fp4 forward calls `blockwise_scale_dequantize`, which is a
  **CPU-only torch reference** (`assert device == cpu`) — it can't trace to a NeuronCore.
- The real device path wants the `blockwise_mm` NKI kernel, which NxD imports from
  `neuronxcc.nki._private.blockwise_mm` — but 2.26 ships it under `_pre_prod_kernels`. Mispathed,
  pre-prod, import fails.
- Swapping my `DenseExperts` for NxD's `ExpertMLPs` (which has fp4 packing) didn't help either: its
  blockwise NKI path `kernel_assert`s, and its `use_torch_block_wise=True` path **crashes torch-xla's
  shape inference on a data-dependent index op.** MoE routing that won't trace — the exact problem the
  dense trick solved for compute, now back for quantization.

**fp4 on inf2 is a wall in this SDK.** I'd spent real time being sure it was the only door. It wasn't
even a door.

## The breakthrough: I was quantizing the wrong thing

Here's the reframe that unlocked it. I'd been staring at the experts because they're huge. But the
experts were *already int8* when I OOM'd. The 3–4 GB I needed wasn't in the experts — it was in the
**weights I'd left in bf16**. Chiefly one: the **tied `lm_head`**, replicated at **1.48 GB per rank**.

The fix wasn't *smaller experts*. It was an **all-int8 squeeze** of everything still fat:

1. **int8 *and* shard the `lm_head`.** `QuantizedColumnParallel(gather_output=True)` slices the 262k-row
   head across ranks and quantizes it: **1.48 GB → ~0.37 GB/rank.** This was the single biggest lever, and
   it had nothing to do with the experts.
2. **int8 the shared dense MLP** (the other half of the dual-path FFN): another ~0.27 GB/rank.
3. Keep the int8 experts.

Per-rank resident dropped to ~12–13 GB — comfortably under 16 GB. And the output was still *right*:

```
Compiler status PASS · MB_TRACED
DEVICE_PARIS True
DEV GEN: 'The capital of France is Paris.'
prefill 232 ms · neff 26.7 GB
```

The 26B MoE ran on a **2-core `inf2.8xlarge`**. The "not achievable without fp4" conclusion was right
that fp4 was out — and wrong about everything else.

## Wall 2: the 8xlarge and the xlarge have the same HBM — but not the same host RAM

Here's the subtlety that trips everyone. `inf2.8xlarge` and `inf2.xlarge` have the **same 2 NeuronCores
and 32 GB HBM**. The difference is **host RAM**: 128 GB vs **16 GB**. My squeeze fit the *device*. Would
it fit the *host* on the cheapest box?

Compiling won't: the ModelBuilder trace holds an fp32 discovery model + a structure model + an fp32
checkpoint dict — **~180 GB peak.** That OOMs even the 8xlarge's 128 GB (I added a 100 GB swapfile to get
the *compile* through, ~40 min). But **compiling and deploying are different jobs.** Deploying a saved
neff doesn't need the model in host RAM at all — the transformer weights live on the device.

So the xlarge deploy is deliberately **slim** (`deploy_sqz.py` / `optb_server_sqz.py`):

- **Structure** (which layers are KV-shared, head dims, sliding window) comes from a **meta-device**
  model instantiation — `accelerate.init_empty_weights()`, ~0 host RAM.
- **Word embeddings** come from a standalone 1.48 GB `embed_tokens.pt` table, loaded once on the host.
- The **neff** carries every transformer weight on the device (`torch.jit.load` + one
  `initialize_with_saved_weights`).
- A **40 GB swapfile** absorbs the one-time neff-load peak.

Result on a stock 16 GB `inf2.xlarge`:

```
NEFF_LOADED in 112s
DEV GEN: 'The capital of France is Paris.'
PREFILL 250 ms
```

**Host RSS peaked ~11 GB; swap was barely touched.** Compile needs 180 GB; deploy needs a laptop's worth.
A 26B model, on a box you'd hesitate to run a 7B on.

## The bug that turned Paris into a wall of spaces

Except the first slim run didn't say Paris. It said:

```
DEV GEN: '                                                              '
DEVICE_PARIS False
```

The neff was proven (it made Paris at compile). So the deploy was feeding it something wrong. Repeated
identical tokens is the fingerprint of **degenerate, near-uniform logits** — the classic symptom of
embeddings at the wrong *magnitude*.

Gemma-4 doesn't use a plain embedding. It uses `Gemma4TextScaledWordEmbedding`, whose forward is:

```python
return super().forward(input_ids) * self.embed_scale   # embed_scale = hidden_size ** 0.5
```

The `×√hidden_size` (= √2816 ≈ **53**) happens *inside the embedding*, and the model forward does **not**
re-normalize — it just does `hidden_states = inputs_embeds`. The neff was traced on *scaled* embeds. My
slim host path used a plain `torch.nn.Embedding` and fed *unscaled* ones — **53× too small.** Everything
downstream washed out to noise.

One line:

```python
ie = emb(ids) * (hidden_size ** 0.5)   # match Gemma4TextScaledWordEmbedding
```

`DEV GEN: 'The capital of France is Paris.'`. This is the same trap that bit the E2B slim server; when you
move embedding lookup to the host, you inherit whatever the model's embedding class was doing.

## Shipping it: 512/128, published, and a clean-pull proof

I recompiled a production `512/128` bucket build (512 total / 128 prompt tokens) — same recipe, ~40 min
on an 8xlarge with swap, neff **26.8 GB**, re-validated on the xlarge (loads in 112 s, ~10 GB host peak,
Paris). Then I wrapped it in a slim OpenAI-compatible server and published:

- **Docker Hub** `xbill9/gemma4-optb-26b:xlarge` — a thin image; the entrypoint pulls the ~28 GB of
  artifacts from the HF repo on first start.
- **HF** `xbill9/gemma-4-26B-A4B-it-inferentia2-xlarge` (public) — neff + embedding table + config +
  server + Dockerfile.

The final proof was the one that matters to anyone but me: a **cold, clean pull.** Fresh spot
`inf2.xlarge`, add swap, `docker run`, walk away.

```
READY in 490.9s — slim int8-squeeze, ModelBuilder TP=2, MAX=512 BUCKET=128
{"role":"assistant","content":"The capital of France is Paris."}
{"response":"AWS Inferentia is a purpose-built machine learning accelerator
             designed to provide high-performance, low-cost inference..."}
```

(No `HF_TOKEN` needed — the repo is public. The 490 s is one-time cold-EBS neff load, then it serves.)
Terminated the box; net ongoing spend, zero.

## The honest caveat

Decode is **~6 tok/s.** That's the price of the `DenseExperts` trick from the last article: computing all
128 experts every token instead of the sparse top-8 — ~16× the necessary expert FLOPs, on 2 cores. It's
*correct* and it's *cheap to host*; it is not *fast*. Making sparse routing actually trace under
ModelBuilder is the real next expedition, and (see the fp4 detour) "the vendor primitive exists" is not
the same as "it traces." For single-user, cost-sensitive, latency-tolerant serving of a 26B on a $0.76/hr
box, the tradeoff is the right one. For a chatbot under load, it isn't yet.

## Takeaways

1. **When something's 3 GB over, don't assume it's the obvious 3 GB.** I burned days on fp4 experts; the
   fix was quantizing the *replicated `lm_head`* I'd never looked at. Profile the residency, don't eyeball
   the architecture.
2. **"Not supported" beats "not documented well."** fp4 has *constants* and *partial code paths* in NxD
   0.19 that make it look one commit away. It isn't — the kernels are CPU-only refs or pre-prod. Read the
   feature guide's supported list and believe it.
3. **Compile-fit and deploy-fit are different budgets.** A model that needs 180 GB of host RAM to *trace*
   can deploy on 16 GB, because the weights live on the device. Slim host loading + swap is what turns a
   24xlarge model into an xlarge product.
4. **Moving embedding lookup to the host means inheriting the embedding *class*.** Gemma's `×√H` scale is
   inside `Gemma4TextScaledWordEmbedding`; miss it and you get a confident wall of whitespace.
5. **Ship the clean-pull proof.** "Works on my validated box" isn't a deliverable; "`docker run` on a
   fresh spot instance says Paris" is.

## Artifacts

- Docker Hub: `xbill9/gemma4-optb-26b:xlarge`
- HF: `xbill9/gemma-4-26B-A4B-it-inferentia2-xlarge` (public)
- Recipe: `tp_mb_moe_sqz.py` (all-int8 squeeze: int8 experts + sharded int8 `lm_head` + int8 shared MLP)
  · `deploy_sqz.py` / `optb_server_sqz.py` (slim host-embedding deploy)

*A 26B mixture-of-experts, on the cheapest accelerator instance AWS rents, for the price of a large coffee
per day. Written with AI assistance in a Claude Code session; every log line quoted is from a real run on
real hardware.*
