"""Static local review GUI. Only explicit downloaded decisions can be imported.

No review server, source edits, auto-acceptance or training approvals.
"""
import argparse
import json
import os
from pathlib import Path
from .dataset_review import read_json,require,safe_file
from .training_export import read_jsonl
from .clear_cutpoint_release import validate,review_ids

PAGE = r'''<!doctype html><meta charset="utf-8"><title>Clear cut-point review</title>
<style>body{font:16px system-ui;background:#182027;color:#eee;margin:20px}button,input,select{font:inherit;margin:5px;padding:7px}canvas{max-width:100%;background:#111}#status{white-space:pre-wrap}label{display:inline-block}textarea{width:90%;height:55px}a{color:#8ee}</style>
<h1>Clear cut-point review ? not a motion authorization</h1>
<p>Original robot-head RGB. Cyan = input query; yellow = nominal cut; magenta = acceptable interval.
Confirm attachment, target identity, visibility and legibility. Toggle marks OFF to judge the RGB itself.
Held-out images require human review. Do not train on this page or overlays.</p>
<button onclick="step(-1)">Previous</button><button onclick="step(1)">Next</button>
<input id="jump" type="number" min="1" style="width:80px"><button onclick="go()">Go</button>
<label><input id="marks" type="checkbox" onchange="draw()">Show diagnostic marks</label>
<div id="status"></div><canvas id="view" width="848" height="408"></canvas>
<details><summary>Query-centred input crop (no diagnostic marks)</summary><img id="crop" width="768"></details>
<p><label>Reviewer <input id="who" placeholder="Your name"></label>
<label>Type <select id="kind"><option>human</option><option>assistant</option></select></label></p>
<textarea id="reason" placeholder="What is clear, ambiguous or incorrect?"></textarea><br>
<button onclick="decide('accept')">Accept this image</button><button onclick="decide('hold')">Hold</button>
<button onclick="decide('reject')">Reject</button><button onclick="download()">Download decisions JSON</button>
<label>Resume from JSON <input type="file" onchange="resume(this.files[0])"></label>
<p>No default decision. Downloads contain only explicitly reviewed images. Keep the downloaded JSON outside the immutable dataset.</p>
<script>
const rows=__ROWS__;let pos=0,decisions={},pic=new Image();const el=id=>document.getElementById(id);
function render(){const r=rows[pos];el('jump').value=pos+1;el('status').textContent=`${pos+1}/${rows.length} | ${r.split} | ${r.id}\n${JSON.stringify(r.legibility)}\nDecision: ${decisions[r.id]?.decision||'pending'} | reviewed ${Object.keys(decisions).length}`;el('reason').value=decisions[r.id]?.reason||'';el('crop').src=r.crop;pic.onload=draw;pic.src=r.rgb;}
function draw(){const c=el('view').getContext('2d'),r=rows[pos];c.clearRect(0,0,848,408);c.drawImage(pic,0,0);if(!el('marks').checked)return;c.lineWidth=2;c.strokeStyle='cyan';c.beginPath();c.arc(...r.query,6,0,2*Math.PI);c.stroke();c.strokeStyle='magenta';c.beginPath();r.interval.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.stroke();c.strokeStyle='yellow';c.beginPath();c.arc(...r.cut,4,0,2*Math.PI);c.stroke();}
function step(d){pos=Math.max(0,Math.min(rows.length-1,pos+d));render();}
function go(){const n=Number(el('jump').value);if(Number.isInteger(n)){pos=Math.max(0,Math.min(rows.length-1,n-1));render();}}
function decide(decision){const reviewer=el('who').value.trim(),reason=el('reason').value.trim();if(!reviewer||!reason){alert('Enter your name and a reason.');return;}const r=rows[pos];decisions[r.id]={id:r.id,rgb_sha256:r.rgb_sha256,reviewer,reviewer_type:el('kind').value,reason,decision};render();}
function download(){const blob=new Blob([JSON.stringify(Object.values(decisions),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='clear_cutpoint_reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
async function resume(file){try{const values=JSON.parse(await file.text()),next={};if(!Array.isArray(values))throw Error('Expected array');for(const r of values){const row=rows.find(x=>x.id===r.id);if(!row||row.rgb_sha256!==r.rgb_sha256||next[r.id]||!['accept','hold','reject'].includes(r.decision))throw Error('Unmatched/duplicate review');next[r.id]=r;}decisions=next;render();}catch(e){alert(e.message);}}
render();
</script>'''


def build(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(root),'New review directory outside release required')
    validate(root,allow_draft=True);rows=list(read_jsonl(root/'index.jsonl'));required=set(review_ids(rows))
    output.mkdir(parents=True);cards=[]
    for r in rows:
        if r['id'] not in required: continue
        label=read_json(safe_file(root,r['files']['label']))
        rel=lambda p:os.path.relpath(safe_file(root,p),output).replace('\\','/')
        cards.append(dict(id=r['id'],split=r['split'],rgb_sha256=r['rgb_sha256'],legibility=r['legibility'],
                          rgb=rel(r['files']['rgb']),crop=rel(r['files']['crop']),query=r['query_pixel_uv'],
                          cut=r['answer']['cut_point_uv'],interval=label['accepted_interval_uv']))
    (output/'index.html').write_text(PAGE.replace('__ROWS__',json.dumps(cards).replace('<','\\u003c')),encoding='utf-8')
    return dict(path=str(output/'index.html'),images=len(cards))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv);print(json.dumps(build(a.dataset,a.output)))


if __name__=='__main__':main()
