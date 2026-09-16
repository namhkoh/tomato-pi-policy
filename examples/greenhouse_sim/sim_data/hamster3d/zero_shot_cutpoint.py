"""Zero-shot 3D HAMSTER on tomato cut-point examples (3D Pointing prompt style).

Uses the released checkpoint unchanged. Input: release RGB (848x408) + native optical-Z depth (metres).
Query and output coordinates are 0-1000 normalized (same convention as our task). Depth output is metres.
Reports pixel error vs the label cut point and the predicted depth vs native depth at the label pixel.
"""
import json, sys, time, re
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0,'/workspace/nhkoh/tomato-vlm/greenhouse_training_code/examples/greenhouse_sim')
from sim_data.training_export import read_jsonl
from hamster3d.inference import Hamster3DPredictor
root=Path('/workspace/nhkoh/tomato-vlm/grounding_release'); out=Path(sys.argv[1]); n_per_class=int(sys.argv[2]) if len(sys.argv)>2 else 4
rows=[r for r in read_jsonl(root/'index.jsonl') if r['split']=='validation']
import random; rng=random.Random(11)
picks=rng.sample([r for r in rows if r['answer']['status']=='localized'],n_per_class)+rng.sample([r for r in rows if r['answer']['status']=='abstain'],n_per_class)

import torch
from hamster3d.inference.preprocessing import prepare_inputs, build_geometry_inputs
def generate_pointing(pred,rgb,depth,user_text,max_new_tokens=256):
    """Same pipeline as Hamster3DPredictor.predict, but with a custom user turn (pointing style)."""
    inputs=prepare_inputs(rgb,depth); rgb_r=inputs['rgb_resized']; depth_r=inputs['depth_resized']
    geo=build_geometry_inputs(rgb_r,depth_r,device=pred.device)
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':user_text}]}]
    text=pred.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    mi=pred.processor(text=[text],images=[Image.fromarray(rgb_r)],padding=True,return_tensors='pt').to(pred.device)
    dt=next(pred.model.parameters()).dtype
    for k,v in list(mi.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v): mi[k]=v.to(dt)
    mi['geometry_encoder_inputs']=[t.to(dt) for t in geo['geometry_encoder_inputs']]; mi['depth_maps']=[t.to(dt) for t in geo['depth_maps']]
    with torch.inference_mode(): out=pred.model.generate(**mi,max_new_tokens=max_new_tokens,do_sample=False,temperature=None,top_p=None)
    return pred.processor.batch_decode(out[:,mi['input_ids'].shape[1]:],skip_special_tokens=True)[0]

pred=Hamster3DPredictor('/workspace/nhkoh/3D_HAMSTER/ckpt', device='cuda:0')
SUFFIX='Report the point_3d location in JSON.'
records=[]
for r in picks:
    rgb=Image.open(root/r['files']['rgb']).convert('RGB'); depth=np.load(root/r['files']['depth']).astype(np.float32)
    valid=np.asarray(Image.open(root/r['files']['validity']))==255; depth=np.where(valid,depth,0.0).astype(np.float32)
    qx,qy=r['query_pixel_uv']; qn=(1000*qx/848,1000*qy/408)
    instr=(f'The target tomato petiole passes through the point ({qn[0]:.0f}, {qn[1]:.0f}). Trace it to its junction with the main stem. '
           f'Point to the nominal cut point 10 mm along the petiole from that junction, if the cut region is visible.')
    t=time.time()
    # Use the predictor's pipeline but with the pointing prompt: build messages manually via prompt_style hook if available
    try:
        raw=generate_pointing(pred,rgb,depth,instr+'\n'+SUFFIX)
    except Exception as e:
        raw=f'ERROR {type(e).__name__}: {e}'
    lat=time.time()-t
    m=re.search(r'```json\s*(.*?)\s*```',raw,re.DOTALL); pts=None
    try:
        arr=json.loads(m.group(1) if m else raw); pts=[e.get('point_3d') or e.get('point_2d') for e in arr if isinstance(e,dict)]
    except Exception: pass
    rec=dict(id=r['id'],truth=r['answer'],query_px=[qx,qy],raw=raw,points=pts,latency_s=lat)
    if pts and pts[0] and len(pts[0])>=2:
        u,v=pts[0][0]*848/1000,pts[0][1]*408/1000; rec['pred_px']=[u,v]
        if r['answer']['cut_point_uv']: rec['err_px']=float(np.hypot(u-r['answer']['cut_point_uv'][0],v-r['answer']['cut_point_uv'][1]))
        if len(pts[0])>=3:
            rec['pred_depth_m']=pts[0][2]; px,py=int(min(max(u,0),847)),int(min(max(v,0),407)); rec['native_depth_at_pred_m']=float(depth[py,px]) if valid[py,px] else None
            rec['native_depth_at_query_m']=float(depth[int(qy),int(qx)]) if valid[int(qy),int(qx)] else None
    records.append(rec); print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True); print('   RAW:',raw[:300].replace('\n',' '),flush=True)
out.write_text(json.dumps(records,indent=1)); print('wrote',out)
