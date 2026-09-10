"use strict";
const $ = id => document.getElementById(id);
let state, current, saving=false, loaded=false, viewed=new Set(), kind='rgb';
const selected = () => state?.samples.find(s=>s.id===current);
function notice(text,error=false){$('notice').textContent=text;$('notice').classList.toggle('error',error);}
function filtered(){const f=$('filter').value;return (state?.samples||[]).filter(s=>f==='all'||(f==='pending'?!s.human_review:s.kind===f));}
function buttons(){
  const s=selected(), ready=s&&!saving&&!s.human_review&&loaded&&['rgb','mask','depth'].every(k=>viewed.has(k))&&$('inspected').checked&&$('reviewer').value.trim()&&$('notes').value.trim().length>=20;
  for(const b of document.querySelectorAll('[data-decision]'))b.disabled=!ready||(b.dataset.decision==='accept'&&s.source_blocks.length>0);
  for(const id of ['filter','sample','prev','next','refresh'])$(id).disabled=saving;
  for(const id of ['notes','inspected'])$(id).disabled=saving||!s||Boolean(s.human_review);
  for(const b of document.querySelectorAll('[data-view]'))b.disabled=saving||!s;
  $('help').textContent=s?.human_review?'Recorded human decisions are read-only.':'Load all three views, enter your name and note, then explicitly confirm inspection.';
}
function marker(tag,attributes){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attributes))el.setAttribute(k,String(v));$('overlay').append(el);}
function markers(){
  $('overlay').replaceChildren();$('overlay').toggleAttribute('hidden',!$('markers').checked||kind==='depth');
  const s=selected();if(!s)return;
  const q=s.input.query_pixel_uv, r=s.input.region_xyxy;
  if(q)marker('circle',{cx:q[0],cy:q[1],r:5,fill:'none',stroke:'cyan','stroke-width':1.5});
  if(r)marker('rect',{x:r[0],y:r[1],width:r[2]-r[0],height:r[3]-r[1],fill:'none',stroke:'cyan','stroke-width':1.5});
  const cut=s.answer.cut_point_uv;
  if(cut){marker('path',{d:`M ${cut[0]-5} ${cut[1]} h 10 M ${cut[0]} ${cut[1]-5} v 10`,stroke:'white','stroke-width':2,fill:'none'});}
}
function view(next){
  const s=selected();if(!s||saving)return;kind=next;loaded=false;
  for(const b of document.querySelectorAll('[data-view]'))b.classList.toggle('selected',b.dataset.view===kind);
  const expected=s.images[kind], sid=s.id, requestedKind=kind;
  $('evidence').onload=()=>{if(current===sid&&$('evidence').getAttribute('src')===expected){loaded=true;viewed.add(requestedKind);buttons();}};
  $('evidence').onerror=()=>{loaded=false;buttons();notice('Evidence failed to load or its binding changed.',true);};
  $('evidence').src=expected;
  $('caption').textContent=kind==='rgb'?'Original full 848×408 robot-head RGB. Markers are optional and review-only.':kind==='mask'?'Exact native target/foreground component highlighted in green. Native identity does not prove RGB readability.':'Saved Isaac optical-Z values, coloured on a fixed 0.04–2.00 m scale. No replacement depth is computed.';
  markers();buttons();
}
function render(preferred){
  const list=filtered();current=list.some(s=>s.id===preferred)?preferred:list[0]?.id;
  $('sample').replaceChildren();for(const s of list){const o=document.createElement('option');o.value=s.id;o.textContent=`${s.family} · ${s.kind} · ${s.human_review?.decision||'pending'}`;$('sample').append(o);}
  $('progress').textContent=`${state.human_reviewed}/${state.total} human-reviewed · ${state.assistant_reviewed} assistant assessments · no motion episodes`;
  $('notes').value='';$('inspected').checked=false;$('markers').checked=false;viewed=new Set();loaded=false;
  const s=selected();
  if(!s){for(const id of ['kind','identity','instruction','answer','saved-status','caption'])$(id).textContent='';$('kind').textContent='No examples in this filter';$('evidence').removeAttribute('src');$('overlay').replaceChildren();$('warning').hidden=true;$('advice-panel').hidden=true;buttons();return;}
  $('sample').value=current;$('kind').textContent=s.kind;$('identity').textContent=s.id+' · '+s.split;
  $('instruction').textContent=s.input.instruction;$('answer').textContent=JSON.stringify(s.answer,null,2);
  $('saved-status').textContent=s.human_review?`${s.human_review.decision} · human: ${s.human_review.reviewer}`:'Pending new-task human review';
  $('notes').value=s.human_review?.notes||'';
  $('warning').hidden=!s.source_blocks.length;$('warning').textContent='Source hold prevents acceptance: '+s.source_blocks.map(x=>x.notes).join(' ');
  $('advice-panel').hidden=!s.assistant_review;$('advice').textContent=s.assistant_review?`${s.assistant_review.decision}: ${s.assistant_review.notes}`:'';
  $('records').textContent='Append-only reviews: '+state.records_directory;view('rgb');
}
async function refresh(){if(saving)return;try{const r=await fetch('/api/state'),s=await r.json();if(!r.ok)throw new Error(s.error);if(s.gui_version!=='active_perception_native_review.v1')throw new Error('Wrong reviewer backend');state=s;render(current);}catch(e){notice(e.message,true);}}
async function save(decision){
  const s=selected();if(!s||saving||$(decision).disabled)return;saving=true;buttons();notice('Checking native evidence and saving one human decision…');
  try{const r=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':document.querySelector('meta[name=review-token]').content},body:JSON.stringify({sample_id:s.id,decision,notes:$('notes').value.trim(),reviewer:$('reviewer').value.trim(),inspected:$('inspected').checked,pilot_sha256:state.pilot_sha256})});
    const d=await r.json();if(!r.ok)throw new Error(d.error);state=d.state;saving=false;render(filtered().find(x=>!x.human_review)?.id||s.id);notice(`Saved ${s.id}: ${decision}. No v3 review, source image or depth file was changed.`);
  }catch(e){saving=false;notice(e.message+' Refresh before retrying.',true);buttons();}
}
$('sample').onchange=()=>render($('sample').value);$('filter').onchange=()=>render();$('refresh').onclick=refresh;$('markers').onchange=markers;
for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{if(saving)return;const list=filtered(),i=list.findIndex(s=>s.id===current);if(list.length)render(list[(i+delta+list.length)%list.length].id);};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>view(b.dataset.view);
for(const b of document.querySelectorAll('[data-decision]'))b.onclick=()=>save(b.dataset.decision);
for(const id of ['notes','reviewer','inspected'])$(id).oninput=buttons;
refresh();
