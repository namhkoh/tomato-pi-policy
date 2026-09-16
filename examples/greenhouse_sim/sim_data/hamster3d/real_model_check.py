"""One-GPU real-model check: forward with labels (finite loss), backward through geometry merger, generate, probe prefix stability."""
import sys, json, torch, time
sys.path.insert(0,'agro')
import cutpoint_data as cd
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from hamster3d.model import register_qwen3_vl_geometry
register_qwen3_vl_geometry()
root=Path('/workspace/nhkoh/tomato-vlm/grounding_release'); ck='ckpt'
proc=AutoProcessor.from_pretrained(ck,trust_remote_code=True)
model=AutoModelForImageTextToText.from_pretrained(ck,dtype=torch.bfloat16,trust_remote_code=True,attn_implementation='sdpa').to('cuda')
model.config.use_cache=False
for n,p in model.named_parameters(): p.requires_grad=not n.startswith('model.geometry_encoder.')
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.train()
print('gradient checkpointing:',model.is_gradient_checkpointing)
index={r['id']:r for r in cd.read_jsonl(root/'index.jsonl')}; train=[index[r['id']] for r in cd.read_jsonl(root/'splits/train.jsonl')]
rows=[next(r for r in train if r['answer']['status']==s) for s in ('localized','abstain')]
dt=torch.bfloat16
for row in rows:
    enc=cd.encode(row,root,proc,640); w=cd.token_weights(enc['labels'],proc.tokenizer,row,None,3.)
    inputs={k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in enc.items()}
    inputs['geometry_encoder_inputs']=[t.to('cuda',dt) for t in inputs['geometry_encoder_inputs']]; inputs['depth_maps']=[t.to('cuda',dt) for t in inputs['depth_maps']]
    inputs['pixel_values']=inputs['pixel_values'].to(dt)
    torch.cuda.reset_peak_memory_stats(); t=time.time()
    out=model(**inputs); loss=out.loss; loss.backward()
    gm=[p.grad for n,p in model.named_parameters() if 'geometry_merger' in n and p.grad is not None]
    gl=[p.grad for n,p in model.named_parameters() if 'language_model.layers.35' in n and p.grad is not None]
    print(f"{row['answer']['status']}: loss={loss.item():.4f} finite={torch.isfinite(loss).item()} | grads: merger tensors={len(gm)} nonzero={any(bool((g!=0).any()) for g in gm)} finite={all(bool(torch.isfinite(g).all()) for g in gm)} | LM last layer grads={len(gl)} | peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB | {time.time()-t:.1f}s")
    model.zero_grad(set_to_none=True)
# generation with the untrained checkpoint (pipeline sanity)
model.eval(); model.config.use_cache=True
row=rows[0]; rgb,depth,valid=cd.load_frame(root,row); img,geo,dmap=cd.prepare(rgb,depth,640)
text=proc.apply_chat_template(cd.messages(row),tokenize=False,add_generation_prompt=True)
mi=proc(text=[text],images=[img],return_tensors='pt').to('cuda'); mi['pixel_values']=mi['pixel_values'].to(dt); mi['geometry_encoder_inputs']=[geo.to('cuda',dt)]; mi['depth_maps']=[dmap.to('cuda',dt)]
with torch.inference_mode(): o=model.generate(**mi,max_new_tokens=96,do_sample=False,temperature=None,top_p=None)
raw=proc.batch_decode(o[:,mi['input_ids'].shape[1]:],skip_special_tokens=True)[0]; print('untrained generate:',raw.replace('\n',' ')[:200],'| parse:',cd.parse_output(raw)[2] or cd.parse_output(raw)[0]['status'])
# probe prefix stability
pref='```json\n[{"point_3d": '
for val in ('[','null'):
    a=proc(text=[text+pref],images=[img],return_tensors='pt')['input_ids']; b=proc(text=[text+pref+val],images=[img],return_tensors='pt')['input_ids']
    print(f"probe prefix stable for {val!r}: {torch.equal(b[:,:a.shape[1]],a)} (+{b.shape[1]-a.shape[1]} tok)")
print('REAL_MODEL_CHECK_OK')
