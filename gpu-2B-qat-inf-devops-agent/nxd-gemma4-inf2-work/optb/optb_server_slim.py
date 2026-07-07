"""Persistent Gemma-4-E2B inference server on Inferentia2 (Option B, two-graph KV-cache).
SLIM-HOST variant: loads ONLY the embedding + PLE tables on the host (in bf16), NOT the 35
transformer decoder blocks (those are baked into the neffs and never used host-side). This drops
host RAM from ~20 GB (full fp32 model) to ~6 GB so it fits inf2.xlarge (16 GiB) without swapping.

Everything downstream (neffs, sampling, routes) is identical to optb_server.py.
Env: HOST_DTYPE (bf16|fp32, default bf16), SELFTEST=1 to run a France->Paris parity check at boot.
NOTE: no authentication (per request). Bound to 0.0.0.0."""
import os
os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"
import sys, json, time, threading, glob, resource, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "128"))
BUCKET = int(os.environ.get("KV_BUCKET", "32"))
PRE_NEFF = os.environ.get("KV_PRE_OUT", "/workspace/kv_pre_neff.pt")
DEC_NEFF = os.environ.get("KV_DEC_OUT", "/workspace/kv_dec_neff.pt")
HOST_DTYPE = torch.bfloat16 if os.environ.get("HOST_DTYPE", "bf16") == "bf16" else torch.float32
SELFTEST = os.environ.get("SELFTEST", "0") == "1"
NEG = torch.finfo(torch.float32).min
NEG_INF = float("-inf")
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = "gemma-4-E2B-it"

def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2)  # KB->GB on linux

from transformers import AutoTokenizer, AutoConfig, Gemma4ForConditionalGeneration
from safetensors import safe_open
import torch_neuronx
print(f"loading tokenizer + SLIM host embeddings (dtype={HOST_DTYPE}, no transformer weights)...", flush=True)
tok = AutoTokenizer.from_pretrained(MP)
cfg_full = AutoConfig.from_pretrained(MP)
# 1) build the whole model on META (allocates NO memory)
with torch.device("meta"):
    m = Gemma4ForConditionalGeneration(cfg_full)
m.eval()
# 2) load ONLY language-model params that are NOT decoder layers (embeddings + PLE + projections/norms),
#    straight from the safetensors shards -> the 35 transformer blocks are never materialized.
shards = sorted(glob.glob(os.path.join(MP, "*.safetensors")))
if not shards:
    raise RuntimeError(f"no *.safetensors in {MP}")
sd, nbytes = {}, 0
for st in shards:
    with safe_open(st, framework="pt") as f:
        for k in f.keys():
            if ".language_model." in k and ".layers." not in k:
                t = f.get_tensor(k).to(HOST_DTYPE)
                sd[k] = t; nbytes += t.numel() * t.element_size()
missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
lang = m.model.language_model; cfg = lang.config; SW = cfg.sliding_window
# meta-construction leaves non-persistent buffers (not in the safetensors) on the meta device.
# Materialize the ones the HOST path touches; embed_tokens.embed_scale = sqrt(hidden_size).
# (buffers under the decoder layers run only on-device via the neffs, so they're irrelevant here.)
for _name, _buf in list(lang.named_buffers()):
    if "layers." in _name or not _buf.is_meta:
        continue
    _mod = lang; *_parents, _leaf = _name.split(".")
    for _p in _parents: _mod = getattr(_mod, _p)
    if _leaf == "embed_scale":
        _mod.register_buffer(_leaf, torch.tensor(float(cfg.hidden_size) ** 0.5), persistent=False)
    else:
        print(f"WARN: unmaterialized host buffer {_name} -> zeros", flush=True)
        _mod.register_buffer(_leaf, torch.zeros(tuple(_buf.shape)), persistent=False)
print(f"slim-loaded {len(sd)} tensors ({round(nbytes/1e9,2)} GB) for host embeddings; "
      f"peak RSS so far {rss_gb()} GB", flush=True)
NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        LINFO[i] = (a.k_proj.out_features // a.head_dim, a.head_dim); NONSHARED.append(i)
ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}

print("loading neffs onto NeuronCores...", flush=True)
t0 = time.time()
PRE = torch.jit.load(PRE_NEFF)
DEC = torch.jit.load(DEC_NEFF)
LOCK = threading.Lock()          # one Inferentia device -> serialize requests
_counter = [0]

# ---- /metrics (Prometheus text format, no deps) ----
_START = time.time()
_MLOCK = threading.Lock()
_METRICS = {"requests_total": 0, "errors_total": 0, "prompt_tokens_total": 0,
            "completion_tokens_total": 0, "generation_seconds_total": 0.0, "last_tok_per_s": 0.0}

def _bump(reqs=0, ptoks=0, ctoks=0, secs=0.0, err=0, tps=None):
    with _MLOCK:
        _METRICS["requests_total"] += reqs
        _METRICS["prompt_tokens_total"] += ptoks
        _METRICS["completion_tokens_total"] += ctoks
        _METRICS["generation_seconds_total"] += secs
        _METRICS["errors_total"] += err
        if tps is not None:
            _METRICS["last_tok_per_s"] = tps

def _cur_rss_bytes():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0

def _render_metrics():
    with _MLOCK:
        m = dict(_METRICS)
    up = time.time() - _START
    avg = (m["completion_tokens_total"] / m["generation_seconds_total"]) if m["generation_seconds_total"] > 0 else 0.0
    def g(name, help_, typ, val):
        return f"# HELP {name} {help_}\n# TYPE {name} {typ}\n{name} {val}"
    return "\n".join([
        g("gemma_up", "1 if serving", "gauge", 1),
        g("gemma_uptime_seconds", "seconds since process start", "gauge", f"{up:.1f}"),
        g("gemma_requests_total", "total generation requests", "counter", m["requests_total"]),
        g("gemma_errors_total", "total request errors", "counter", m["errors_total"]),
        g("gemma_prompt_tokens_total", "total prompt tokens", "counter", m["prompt_tokens_total"]),
        g("gemma_completion_tokens_total", "total generated tokens", "counter", m["completion_tokens_total"]),
        g("gemma_generation_seconds_total", "total generation wall seconds", "counter", f"{m['generation_seconds_total']:.3f}"),
        g("gemma_tokens_per_second_last", "tok/s of the most recent request", "gauge", f"{m['last_tok_per_s']:.2f}"),
        g("gemma_tokens_per_second_avg", "avg tok/s over all requests", "gauge", f"{avg:.2f}"),
        g("gemma_max_total_tokens", "configured KV_MAX", "gauge", MAX),
        g("gemma_max_prompt_tokens", "configured KV_BUCKET", "gauge", BUCKET),
        g("process_resident_memory_bytes", "resident set size", "gauge", _cur_rss_bytes()),
    ]) + "\n"

def embed_ids(id_list):
    ids = torch.tensor([id_list])
    with torch.no_grad():
        ie = lang.embed_tokens(ids); ple = lang.get_per_layer_inputs(ids, ie)
    # tables are bf16 (low RAM) but the neffs were traced with fp32 activations -> cast the
    # small per-token tensors to fp32 so ops.neuron.forward_v2 gets the dtype it expects.
    return ie.float(), ple.float()

def host_pos(pos):
    ar = torch.arange(MAX)
    oh = (ar == pos).view(1, 1, MAX, 1).to(torch.float32)
    v = ar <= pos
    f = torch.where(v, 0.0, NEG).view(1, 1, 1, MAX)
    s = torch.where(v & (ar > pos - SW), 0.0, NEG).view(1, 1, 1, MAX)
    return torch.tensor([[pos]], dtype=torch.long), oh, f, s

def pick(logits, temperature, top_k, top_p):
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(logits))
    logits = logits.float() / float(temperature)
    if top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, NEG_INF), logits)
    probs = torch.softmax(logits, dim=-1)
    if top_p and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum - sp <= top_p
        sp = torch.where(keep, sp, torch.zeros_like(sp))
        probs = torch.zeros_like(probs).scatter(0, si, sp)
    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(logits))
    probs = probs / total
    return int(torch.multinomial(probs, 1))

def generate_ids(prompt_ids, max_new, temperature, top_k, top_p, stop_ids):
    n0 = len(prompt_ids)
    if n0 > BUCKET:
        raise ValueError(f"prompt {n0} tokens > BUCKET {BUCKET}")
    pad = prompt_ids + [0] * (BUCKET - n0)
    ie, ple = embed_ids(pad)
    am = torch.tensor([[1] * n0 + [0] * (BUCKET - n0)])
    lg, ks, vs = PRE(ie, am, ple)
    first = pick(lg[0, n0 - 1], temperature, top_k, top_p)
    key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
    val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
    seq = [first]; cur = n0
    finish = "length"
    cap = min(max_new, MAX - n0 - 1)
    for _ in range(cap):
        if seq[-1] in EOS or seq[-1] in stop_ids:
            finish = "stop"; break
        ie1, ple1 = embed_ids([seq[-1]])
        pid, oh, f, s = host_pos(cur)
        lg1, key_bufs, val_bufs = DEC(ie1, ple1, pid, oh, f, s, key_bufs, val_bufs)
        seq.append(pick(lg1[0, 0], temperature, top_k, top_p)); cur += 1
    gen = [t for t in seq if t not in EOS and t not in stop_ids]
    return gen, n0, finish

def run_chat(messages, max_new, temperature, top_k, top_p, stop):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    stop_ids = set()
    for s in (stop or []):
        try: stop_ids.update(tok.encode(s, add_special_tokens=False))
        except Exception: pass
    t_req = time.time()
    with LOCK:
        gen, n0, finish = generate_ids(prompt_ids, max_new, temperature, top_k, top_p, stop_ids)
    text = tok.decode(gen, skip_special_tokens=True)
    dt = time.time() - t_req; ct = len(gen)
    _bump(reqs=1, ptoks=n0, ctoks=ct, secs=dt, tps=(ct / dt if dt > 0 else 0.0))
    return text, n0, ct, finish

# warm up
with LOCK:
    try: _ = generate_ids(tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist(), 3, 0.0, 0, 1.0, set())
    except Exception as e: print("warmup:", e, flush=True)
print(f"READY in {round(time.time()-t0,1)}s — serving on :{PORT} (SLIM host, peak RSS {rss_gb()} GB)", flush=True)

if SELFTEST:
    txt, pt, ct, fin = run_chat([{"role": "user", "content": "What is the capital of France? Answer in one word."}], 8, 0.0, 0, 1.0, None)
    print(f"SELFTEST capital-of-France -> {txt!r}  (expect 'Paris')  peakRSS={rss_gb()}GB", flush=True)

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _send_text(self, code, text):
        b = text.encode()
        self.send_response(code); self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")
    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local-inferentia2"}]})
        elif p in ("/health", "/ping"):
            self._send(200, {"status": "ok"})
        elif p == "/metrics":                     # Prometheus metrics
            self._send_text(200, _render_metrics())
        else:
            self._send(200, {"status": "ok", "model": f"{MODEL_NAME} (Option B / torch_neuronx, slim-host)",
                             "device": "Inferentia2", "max_total_tokens": MAX, "max_prompt_tokens": BUCKET,
                             "routes": ["/generate", "/v1/chat/completions", "/v1/completions", "/v1/models", "/health", "/metrics"]})
    def do_POST(self):
        try:
            path = self.path.rstrip("/")
            body = self._body()
            if path == "/v1/chat/completions":
                msgs = body.get("messages")
                if not msgs: return self._send(400, {"error": {"message": "missing 'messages'"}})
                temp = body.get("temperature", 0.7)
                text, pt, ct, finish = run_chat(
                    msgs, int(body.get("max_tokens", 256)), temp,
                    int(body.get("top_k", 0)), float(body.get("top_p", 0.95)), body.get("stop"))
                _counter[0] += 1
                self._send(200, {
                    "id": f"chatcmpl-{int(time.time())}-{_counter[0]}", "object": "chat.completion",
                    "created": int(time.time()), "model": body.get("model", MODEL_NAME),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": finish}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
            elif path == "/v1/completions":
                prompt = body.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
                if not prompt: return self._send(400, {"error": {"message": "missing 'prompt'"}})
                temp = body.get("temperature", 0.7)
                text, pt, ct, finish = run_chat(
                    [{"role": "user", "content": prompt}], int(body.get("max_tokens", 256)), temp,
                    int(body.get("top_k", 0)), float(body.get("top_p", 0.95)), body.get("stop"))
                _counter[0] += 1
                self._send(200, {
                    "id": f"cmpl-{int(time.time())}-{_counter[0]}", "object": "text_completion",
                    "created": int(time.time()), "model": body.get("model", MODEL_NAME),
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
            elif path == "/generate":
                prompt = body.get("prompt", "")
                if not prompt: return self._send(400, {"error": "missing 'prompt'"})
                text, pt, ct, finish = run_chat(
                    [{"role": "user", "content": prompt}], int(body.get("max_new_tokens", 110)),
                    body.get("temperature", 0.0), int(body.get("top_k", 0)),
                    float(body.get("top_p", 1.0)), body.get("stop"))
                self._send(200, {"prompt": prompt, "response": text, "prompt_tokens": pt,
                                 "gen_tokens": ct, "finish_reason": finish})
            else:
                self._send(404, {"error": {"message": f"unknown route {path}"}})
        except ValueError as e:
            _bump(err=1)
            self._send(400, {"error": {"message": str(e)}})
        except Exception as e:
            _bump(err=1)
            self._send(500, {"error": {"message": repr(e)}})
    def log_message(self, *a): pass

ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
