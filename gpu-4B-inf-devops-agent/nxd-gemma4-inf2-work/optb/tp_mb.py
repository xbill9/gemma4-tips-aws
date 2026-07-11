"""Path B: TP=2 prefill+decode via NxD ModelBuilder — ONE weight set shared across both buckets
(sharded once from a checkpoint_loader, so fp32 fits ~8GB/rank AND stays correct). Reuses the
Path-A unified forward + one-hot-scatter KV write. module_cls builds parallel-layer STRUCTURE only
(no weight copy); ModelBuilder's get_sharded_checkpoint fills + shares the weights."""
import sys, os, types, time
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
import multiprocessing
MP="/workspace/real-gemma4-E4B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
CARGS="--model-type transformer --auto-cast all --auto-cast-type bf16"

def _kv_rank_width(nkv): return nkv//TP if nkv%TP==0 else nkv

def _discover():
    """Load ref model on host once to get NONSHARED/LINFO/softcap/SW (structure metadata)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,cfg.sliding_window

def build_module():
    """Build the sharded Gemma4 Wrap: parallel-layer STRUCTURE only (weights come from checkpoint)."""
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    mm,lang,lm_head,softcap,NONSHARED,LINFO,SW=_discover()
    class GeluTanh(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()
    def col(o): return ColumnParallelLinear(o.in_features,o.out_features,bias=False,gather_output=False)
    def row(o): return RowParallelLinear(o.in_features,o.out_features,bias=False,input_is_parallel=True)
    for lyr in lang.layers[:lang.config.num_hidden_layers]:
        a=lyr.self_attn; hd=a.head_dim
        a.q_proj=col(a.q_proj); a.o_proj=row(a.o_proj)
        if getattr(a,"k_proj",None) is not None:
            nkv=a.k_proj.out_features//hd
            if nkv%TP==0:
                a.k_proj=col(a.k_proj)
                if getattr(a,"v_proj",None) is not None: a.v_proj=col(a.v_proj)
            else:
                a.num_key_value_groups=(a.q_proj.out_features//hd)//nkv
        mp=lyr.mlp; mp.gate_proj=col(mp.gate_proj); mp.up_proj=col(mp.up_proj); mp.down_proj=row(mp.down_proj)
    lang.embed_tokens=torch.nn.Embedding(2, lang.config.hidden_size)   # dummy: embeddings computed on host, not on device
    NK=len(NONSHARED); W={i:_kv_rank_width(LINFO[i][0]) for i in NONSHARED}
    def _sc(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    class ScatterKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def _wr(s,buf,x):
            oh=s.oh[0,0]; scat=torch.einsum('bhsd,ms->bhmd',x,oh); r=oh.sum(-1).view(1,1,MAX,1); return buf*(1.0-r)+scat
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s._wr(s.key[idx],k); s.val[idx]=s._wr(s.val[idx],v); return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lang=lang; s.head=lm_head
            s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
            s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
        def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask):
            cache=ScatterKV(list(s.kbuf),list(s.vbuf),onehot)
            out=s.lang(inputs_embeds=ie,per_layer_inputs=ple,position_ids=position_ids,
                       attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask},use_cache=True,past_key_values=cache)
            lg=_sc(s.head(out.last_hidden_state)); ks,vs=cache.export(); return (lg,)+tuple(ks)+tuple(vs)
    w=Wrap().eval()
    aliases={}
    for j in range(NK): aliases[w.kbuf[j]]=1+j
    for j in range(NK): aliases[w.vbuf[j]]=1+NK+j
    return w, aliases


def checkpoint_loader():
    """Full fp32 state dict keyed to match the Wrap module (lang.* + head.*)."""
    import torch
    mm,lang,lm_head,_sc,_ns,_li,_sw=_discover()
    sd={}
    for k,v in lang.state_dict().items():
        if "embed_tokens" in k: continue
        sd["lang."+k]=v.to(torch.float32)
    for k,v in lm_head.state_dict().items(): sd["head."+k]=v.to(torch.float32)
    return sd

def _inputs(lang, ids, SW, NEG, positions):
    import torch
    ids_t=torch.tensor([ids])
    with torch.no_grad(): ie=lang.embed_tokens(ids_t); ple=lang.get_per_layer_inputs(ids_t,ie)
    seq=len(positions); ar=torch.arange(MAX); pos=torch.tensor([positions],dtype=torch.long)
    oh=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.float32)
    q=pos.view(seq,1); mm2=ar.view(1,MAX)
    full=torch.where(mm2<=q,0.0,NEG).view(1,1,seq,MAX)
    slide=torch.where((mm2<=q)&(mm2>q-SW),0.0,NEG).view(1,1,seq,MAX)
    return ie,ple,pos,oh,full,slide

def main():
    import torch
    from transformers import AutoTokenizer
    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    _mm,rlang,_h,_s,NONSHARED,LINFO,SW=_discover()
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt); assert n0<=BUCKET
    pre_in=_inputs(rlang, prompt+[0]*(BUCKET-n0), SW, NEG, list(range(BUCKET)))
    dec_in=_inputs(rlang, [prompt[-1]], SW, NEG, [n0])
    class GemmaInstance(BaseModelInstance):
        def __init__(self): self.module=None; self.input_output_aliases=[{}]
        def load_module(self):                       # called by ModelBuilder UNDER the rank TP context
            self.module, aliases = build_module()    # -> ColumnParallelLinear now actually shards
            self.input_output_aliases=[aliases]
        def get(self, bucket_rank, **kwargs): return self.module, self.input_output_aliases[0]
    inst=GemmaInstance()
    print("building ModelBuilder ...",flush=True)
    mb=ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=checkpoint_loader, compiler_workdir="/workspace/mb_wd")
    mb.add("prefill", inst, [pre_in], compiler_args=CARGS)
    mb.add("decode",  inst, [dec_in], compiler_args=CARGS)
    print("tracing ...",flush=True)
    model=mb.trace(initialize_model_weights=True)
    print("MB_TRACED",flush=True)
    try:
        from neuronx_distributed.trace import parallel_model_save
        parallel_model_save(model,"/workspace/tpmb"); print("MB_SAVED",flush=True)
    except Exception as e: print("save skipped:",e,flush=True)
    print("TPMB_OK",flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()
