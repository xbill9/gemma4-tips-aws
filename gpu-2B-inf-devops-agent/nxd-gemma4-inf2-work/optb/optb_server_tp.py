"""TP=2 + KV-aliasing HTTP server for Gemma4-E2B (~72 tok/s @ 2048, both cores).
Loads serialized parallel neffs (tpa_pre/tpa_dec) via parallel_model_load, seeds per-rank
device-resident KV per request. Endpoints: /health /generate /v1/chat/completions /v1/completions."""
import sys, os, types, time, json, threading
m = types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None
sys.modules["transformers.utils.fx"] = m
import multiprocessing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
MP="/workspace/real-gemma4-E2B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","2048")); BUCKET=int(os.environ.get("KV_BUCKET","512"))
PORT=int(os.environ.get("PORT","8080")); MODEL_NAME="gemma-4-E2B-it"
PRE_DIR=os.environ.get("TPA_PRE","/workspace/tpa_pre"); DEC_DIR=os.environ.get("TPA_DEC","/workspace/tpa_dec")
LOCK=threading.Lock(); READY=threading.Event()

def sp_now():
    # Fetch the CURRENT per-rank aliased KV state params. Must be read AFTER the decode
    # model's first forward: ParallelModel._load() calls move_trace_to_device(), which
    # REPLACES each states._parameters[name] with a NEW privateuseone (device) tensor. Params
    # captured before that first forward are orphaned CPU tensors — writing to them never
    # reaches the device (the req1-ok / req2+-leak bug). Re-reading here returns the live
    # device-resident params so the per-request reseed actually clears prior KV.
    out=[]
    for r in range(TP):
        s=[(int(''.join(c for c in n if c.isdigit())),p) for n,p in dec.models[r].named_parameters() if n.startswith("states.")]
        out.append([p for _,p in sorted(s,key=lambda x:x[0])])
    return out

def boot():
    global torch, tok, lang, NONSHARED, LINFO, SW, EOS, pre, dec, NK, NEG
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    from neuronx_distributed.trace import parallel_model_load
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config; SW=cfg.sliding_window
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    NK=len(NONSHARED)
    ec=mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    t=time.time(); pre=parallel_model_load(PRE_DIR); dec=parallel_model_load(DEC_DIR)
    # Warm up: the first pre()/dec() forward triggers ParallelModel._load() +
    # move_trace_to_device (the ~45s on-device graph load) AND swaps state params to device.
    # Doing it here (not on the first user request) keeps request latency flat and makes
    # sp_now() return device-resident params from request #1 onward.
    warm=tok.apply_chat_template([{"role":"user","content":"Hi"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"][0].tolist()
    generate(warm,3)
    print(f"READY in {round(time.time()-t,1)}s (load+warmup) — TP+alias, MAX={MAX} BUCKET={BUCKET}",flush=True); READY.set()

def embed_ids(idl):
    ids=torch.tensor([idl])
    with torch.no_grad(): ie=lang.embed_tokens(ids); ple=lang.get_per_layer_inputs(ids,ie)
    return ie,ple
def host_pos(pos):
    ar=torch.arange(MAX); oh=(ar==pos).view(1,1,MAX,1).to(torch.float32); valid=ar<=pos
    fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
    return torch.tensor([[pos]],dtype=torch.long),oh,fm,sm

def generate(prompt_ids, max_new):
    prompt=prompt_ids[:BUCKET]; n0=len(prompt)
    pad=prompt+[0]*(BUCKET-n0); ie,ple=embed_ids(pad); am=torch.tensor([[1]*n0+[0]*(BUCKET-n0)])
    lg,ks,vs=pre(ie,am,ple); first=int(lg[0,n0-1].argmax())
    SP=sp_now()  # live device-resident params (see sp_now docstring) — reseed clears prior KV
    for r in range(TP):
        for j in range(NK):
            sk=torch.zeros(1,LINFO[NONSHARED[j]][0],MAX,LINFO[NONSHARED[j]][1]); sk[:,:,:n0,:]=ks[j][:,:,:n0,:]
            sv=torch.zeros(1,LINFO[NONSHARED[j]][0],MAX,LINFO[NONSHARED[j]][1]); sv[:,:,:n0,:]=vs[j][:,:,:n0,:]
            SP[r][j].data.copy_(sk); SP[r][NK+j].data.copy_(sv)
    seq=[first]; cur=n0
    cap=min(max_new, MAX-n0-1)
    for _ in range(max(cap,0)):
        if seq[-1] in EOS: break
        e1,p1=embed_ids([seq[-1]]); pi,o,f,s=host_pos(cur)
        r=dec(e1,p1,pi,o,f,s); lg1=r[0] if isinstance(r,(tuple,list)) else r
        seq.append(int(lg1[0,0].argmax())); cur+=1
    return [x for x in seq if x not in EOS], n0

def run(prompt_ids, max_new):
    t=time.time()
    with LOCK: gen,n0=generate(prompt_ids,max_new)
    dt=time.time()-t; txt=tok.decode(gen,skip_special_tokens=True)
    print(f"[req] pt={n0} ct={len(gen)} {round(len(gen)/dt,1) if dt>0 else 0}tok/s {round(dt,2)}s",flush=True)
    return txt,n0,len(gen)

class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,c,o): b=json.dumps(o).encode(); s.send_response(c); s.send_header("Content-Type","application/json"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def _body(s): n=int(s.headers.get("Content-Length",0)); return json.loads(s.rfile.read(n) or b"{}")
    def do_GET(s):
        p=s.path.rstrip("/")
        if p in ("/health","/ping"): s._j(200 if READY.is_set() else 503, {"status":"ok" if READY.is_set() else "loading"})
        elif p=="/v1/models": s._j(200,{"object":"list","data":[{"id":MODEL_NAME,"object":"model"}]})
        else: s._j(200,{"status":"ok","model":f"{MODEL_NAME} (TP2+alias)","device":"Inferentia2","max_total_tokens":MAX,"max_prompt_tokens":BUCKET,"tps":"~72","routes":["/generate","/v1/chat/completions","/v1/completions","/health"]})
    def do_POST(s):
        path=s.path.rstrip("/")
        if not READY.is_set(): return s._j(503,{"error":{"message":"loading"}})
        try:
            b=s._body()
            if path=="/v1/chat/completions":
                msgs=b.get("messages");
                if not msgs: return s._j(400,{"error":{"message":"missing messages"}})
                enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors="pt",return_dict=True); pids=enc["input_ids"][0].tolist()
                mx=int(b.get("max_tokens",256)); txt,n0,ct=run(pids,mx)
                return s._j(200,{"id":"cmpl","object":"chat.completion","model":MODEL_NAME,"choices":[{"index":0,"message":{"role":"assistant","content":txt},"finish_reason":"stop"}],"usage":{"prompt_tokens":n0,"completion_tokens":ct}})
            elif path=="/v1/completions":
                pr=b.get("prompt",""); enc=tok.apply_chat_template([{"role":"user","content":pr}],add_generation_prompt=True,return_tensors="pt",return_dict=True); pids=enc["input_ids"][0].tolist()
                mx=int(b.get("max_tokens",256)); txt,n0,ct=run(pids,mx)
                return s._j(200,{"id":"cmpl","object":"text_completion","model":MODEL_NAME,"choices":[{"index":0,"text":txt,"finish_reason":"stop"}],"usage":{"prompt_tokens":n0,"completion_tokens":ct}})
            elif path=="/generate":
                pr=b.get("prompt",""); enc=tok.apply_chat_template([{"role":"user","content":pr}],add_generation_prompt=True,return_tensors="pt",return_dict=True); pids=enc["input_ids"][0].tolist()
                mx=int(b.get("max_tokens",256)); txt,n0,ct=run(pids,mx)
                return s._j(200,{"response":txt,"prompt_tokens":n0,"gen_tokens":ct,"finish_reason":"stop"})
            else: return s._j(404,{"error":{"message":f"unknown route {path}"}})
        except Exception as e:
            return s._j(500,{"error":{"message":str(e)}})

def main():
    httpd=ThreadingHTTPServer(("0.0.0.0",PORT),H)
    threading.Thread(target=boot,daemon=True).start()
    print(f"HTTP listening on :{PORT} (loading neffs...)",flush=True); httpd.serve_forever()

if __name__ == "__main__":
    if os.environ.get("_TP_CHILD") != "1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn", force=True); main()
