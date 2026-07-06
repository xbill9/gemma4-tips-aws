"""Persistent Gemma-4-E2B inference server on Inferentia2 (Option B, two-graph KV-cache).
Loads both neffs ONCE, then serves fast (~44 tok/s). Stdlib http.server, no deps.
Routes:
  GET  /                     health
  GET  /v1/models            OpenAI-style model list
  POST /generate             {"prompt","max_new_tokens","temperature","top_k","top_p","stop"}
  POST /v1/chat/completions  OpenAI-compatible (messages[], temperature, top_p, max_tokens, stop)
Sampling: temperature<=0 => greedy; else temperature/top_k/top_p nucleus sampling.
NOTE: no authentication (per request). Bound to 0.0.0.0."""
import os
os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"
import sys, json, time, threading, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MP = "/workspace/real-gemma4-E2B-it"
MAX = int(os.environ.get("KV_MAX", "128"))
BUCKET = int(os.environ.get("KV_BUCKET", "32"))
PRE_NEFF = os.environ.get("KV_PRE_OUT", "/workspace/kv_pre_neff.pt")
DEC_NEFF = os.environ.get("KV_DEC_OUT", "/workspace/kv_dec_neff.pt")
NEG = torch.finfo(torch.float32).min
NEG_INF = float("-inf")
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = "gemma-4-E2B-it"

from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
import torch_neuronx
print("loading tokenizer + model (host embeddings)...", flush=True)
tok = AutoTokenizer.from_pretrained(MP)
m = Gemma4ForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float32, attn_implementation="eager"); m.eval()
lang = m.model.language_model; cfg = lang.config; SW = cfg.sliding_window
NONSHARED, LINFO = [], {}
for i, lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
    a = lyr.self_attn
    if not a.is_kv_shared_layer:
        NONSHARED.append(i); LINFO[i] = (a.k_proj.out_features // a.head_dim, a.head_dim)
ec = m.generation_config.eos_token_id
EOS = set(ec) if isinstance(ec, (list, tuple)) else {ec}

print("loading neffs onto NeuronCores...", flush=True)
t0 = time.time()
PRE = torch.jit.load(PRE_NEFF)
DEC = torch.jit.load(DEC_NEFF)
LOCK = threading.Lock()          # one Inferentia device -> serialize requests
_counter = [0]

def embed_ids(id_list):
    ids = torch.tensor([id_list])
    with torch.no_grad():
        ie = lang.embed_tokens(ids); ple = lang.get_per_layer_inputs(ids, ie)
    return ie, ple

def host_pos(pos):
    ar = torch.arange(MAX)
    oh = (ar == pos).view(1, 1, MAX, 1).to(torch.float32)
    v = ar <= pos
    f = torch.where(v, 0.0, NEG).view(1, 1, 1, MAX)
    s = torch.where(v & (ar > pos - SW), 0.0, NEG).view(1, 1, 1, MAX)
    return torch.tensor([[pos]], dtype=torch.long), oh, f, s

def pick(logits, temperature, top_k, top_p):
    """logits: 1-D tensor over vocab. Returns an int token id."""
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
        keep = cum - sp <= top_p            # keep tokens up to and incl. the one crossing p
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
    with LOCK:
        gen, n0, finish = generate_ids(prompt_ids, max_new, temperature, top_k, top_p, stop_ids)
    text = tok.decode(gen, skip_special_tokens=True)
    return text, n0, len(gen), finish

# warm up
with LOCK:
    try: _ = generate_ids(tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist(), 3, 0.0, 0, 1.0, set())
    except Exception as e: print("warmup:", e, flush=True)
print(f"READY in {round(time.time()-t0,1)}s — serving on :{PORT} (sampling + OpenAI, no auth)", flush=True)

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")
    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local-inferentia2"}]})
        elif p in ("/health", "/ping"):          # vLLM-style liveness
            self._send(200, {"status": "ok"})
        else:
            self._send(200, {"status": "ok", "model": f"{MODEL_NAME} (Option B / torch_neuronx)",
                             "device": "Inferentia2", "max_total_tokens": MAX, "max_prompt_tokens": BUCKET,
                             "routes": ["/generate", "/v1/chat/completions", "/v1/completions", "/v1/models", "/health"]})
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
                # vLLM-style text completion. `prompt` may be a string or a list.
                prompt = body.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0] if prompt else ""
                if not prompt: return self._send(400, {"error": {"message": "missing 'prompt'"}})
                temp = body.get("temperature", 0.7)
                # instruct model -> wrap prompt as a user turn so query_vllm gets coherent output
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
            self._send(400, {"error": {"message": str(e)}})
        except Exception as e:
            self._send(500, {"error": {"message": repr(e)}})
    def log_message(self, *a): pass

ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
