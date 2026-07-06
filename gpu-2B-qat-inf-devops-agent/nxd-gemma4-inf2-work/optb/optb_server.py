"""Persistent Gemma-4-E2B inference server on Inferentia2 (Option B, two-graph KV-cache).
Loads both neffs ONCE (~86s), then serves POST /generate {"prompt": "..."} fast (~44 tok/s).
No external deps (stdlib http.server). Replaces the broken vLLM/NxD endpoint."""
import os
os.environ["NEURON_RT_VISIBLE_CORES"] = "0,1"
import sys, json, time, threading, torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MP = "/workspace/real-gemma4-E2B-it"; MAX = 128; BUCKET = 32
NEG = torch.finfo(torch.float32).min
PORT = int(os.environ.get("PORT", "8080"))

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

print("loading neffs onto NeuronCores (one-time ~86s)...", flush=True)
t0 = time.time()
PRE = torch.jit.load("/workspace/kv_pre_neff.pt")
DEC = torch.jit.load("/workspace/kv_dec_neff.pt")
LOCK = threading.Lock()   # one Inferentia device — serialize requests

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

def generate(prompt, max_new=110):
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids0 = enc["input_ids"][0].tolist(); n0 = len(ids0)
    if n0 > BUCKET:
        return {"error": f"prompt {n0} tokens > BUCKET {BUCKET}; recompile with a larger bucket"}
    pad = ids0 + [0]*(BUCKET-n0)
    ie, ple = embed_ids(pad); am = torch.tensor([[1]*n0 + [0]*(BUCKET-n0)])
    lg, ks, vs = PRE(ie, am, ple)
    first = int(lg[0, n0-1].argmax())
    key_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
    val_bufs = [torch.zeros(1, LINFO[i][0], MAX, LINFO[i][1]) for i in NONSHARED]
    for j in range(len(NONSHARED)):
        key_bufs[j][:, :, :n0, :] = ks[j][:, :, :n0, :]
        val_bufs[j][:, :, :n0, :] = vs[j][:, :, :n0, :]
    seq = [first]; cur = n0; t = time.time()
    for _ in range(min(max_new, MAX - n0 - 1)):
        if seq[-1] in EOS: break
        ie1, ple1 = embed_ids([seq[-1]])
        pid, oh, f, s = host_pos(cur)
        lg1, key_bufs, val_bufs = DEC(ie1, ple1, pid, oh, f, s, key_bufs, val_bufs)
        seq.append(int(lg1[0, 0].argmax())); cur += 1
    dt = time.time() - t
    txt = tok.decode([x for x in seq if x not in EOS], skip_special_tokens=True)
    return {"prompt": prompt, "response": txt, "prompt_tokens": n0,
            "gen_tokens": len(seq), "decode_secs": round(dt, 2),
            "tok_s": round((len(seq)-1)/dt, 1) if dt > 0 else None}

# warm up (triggers device execution paths once)
with LOCK:
    _ = generate("Hi", max_new=3)
print(f"READY in {round(time.time()-t0,1)}s — serving on :{PORT}", flush=True)

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send(200, {"status": "ok", "model": "gemma-4-E2B-it (Option B / torch_neuronx)",
                         "device": "Inferentia2", "max_total_tokens": MAX, "max_prompt_tokens": BUCKET})
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            prompt = body.get("prompt", "")
            if not prompt: return self._send(400, {"error": "missing 'prompt'"})
            with LOCK:
                out = generate(prompt, int(body.get("max_new_tokens", 110)))
            self._send(200, out)
        except Exception as e:
            self._send(500, {"error": repr(e)})
    def log_message(self, *a): pass

ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
