// Read-only v4 browser/evidence check. Never submits a dataset review.
// Start a separate headless Chromium with loopback CDP on port 9223 first.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

const [base='http://127.0.0.1:8882',output]=process.argv.slice(2);
assert(output,'Supply a NEW screenshot directory');
assert.equal(new URL(base).hostname,'127.0.0.1');
await fs.mkdir(output);
const initial=await fetch(base+'/api/state').then(r=>r.json());
assert.equal(initial.gui_version,'active_perception_native_review.v1');
const targets=await fetch('http://127.0.0.1:9223/json').then(r=>r.json());
const ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let seq=0;const pending=new Map(),errors=[],networkErrors=[];
ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);if(!p)return;pending.delete(m.id);clearTimeout(p.timer);if(m.error)p.reject(new Error(JSON.stringify(m.error)));else p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);else if(m.method==='Network.loadingFailed'||(m.method==='Network.responseReceived'&&m.params.response.status>=400))networkErrors.push(m);};
function cdp(method,params={}){return new Promise((resolve,reject)=>{const id=++seq;const timer=setTimeout(()=>{pending.delete(id);reject(new Error('CDP timeout: '+method));},15000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value;}
async function until(expression){for(let i=0;i<120;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,100));}throw new Error('Browser condition timed out: '+expression);}
async function screenshot(name){const r=await cdp('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(output,name),Buffer.from(r.data,'base64'),{flag:'wx'});}
try{
  await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Network.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1450,height:1050,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:base});
  await until('document.querySelector("#evidence")?.naturalWidth===848 && document.querySelector("#sample")?.options.length>0');
  await evaluate('document.querySelector("#filter").value="all";document.querySelector("#filter").dispatchEvent(new Event("change"))');
  const seen=new Set();
  for(const s of initial.samples){
    await evaluate(`document.querySelector('#sample').value=${JSON.stringify(s.id)};document.querySelector('#sample').dispatchEvent(new Event('change'))`);
    await until(`document.querySelector('#evidence').getAttribute('src')===${JSON.stringify(s.images.rgb)} && document.querySelector('#evidence').complete && document.querySelector('#evidence').naturalWidth===848`);
    assert(await evaluate('document.querySelector("#accept").disabled && !document.querySelector("#inspected").checked && !document.querySelector("#markers").checked && getComputedStyle(document.querySelector("#overlay")).display==="none"'));
    await evaluate('document.querySelector("#markers").click()');
    assert(await evaluate('getComputedStyle(document.querySelector("#overlay")).display!=="none"'));
    if(s.kind==='uncertain_region')assert.equal(await evaluate('document.querySelectorAll("#overlay rect").length'),1);
    if(s.answer.cut_point_uv===null)assert.equal(await evaluate('document.querySelectorAll("#overlay path").length'),0);
    const capture=s.kind==='uncertain_region'||!seen.has(s.kind);
    seen.add(s.kind);
    for(const kind of ['rgb','mask','depth']){
      await evaluate(`document.querySelector('[data-view=${kind}]').click()`);
      await until(`document.querySelector('#evidence').getAttribute('src')===${JSON.stringify(s.images[kind])} && document.querySelector('#evidence').complete && document.querySelector('#evidence').naturalWidth===848`);
      assert(await evaluate('document.querySelector("#accept").disabled'));
      if(kind==='depth')assert(await evaluate('getComputedStyle(document.querySelector("#overlay")).display==="none"'));
      if(capture)await screenshot(`${s.family}_${s.kind}_${kind}.png`);
    }
  }
  const final=await fetch(base+'/api/state').then(r=>r.json());
  assert.deepEqual(final.samples.map(s=>[s.human_review,s.assistant_review]),initial.samples.map(s=>[s.human_review,s.assistant_review]));
  assert.equal(errors.length,0,JSON.stringify(errors));
  await fs.writeFile(path.join(output,'browser_check.json'),JSON.stringify({checked_utc:new Date().toISOString(),
    gui_version:initial.gui_version,pilot_sha256:initial.pilot_sha256,total:initial.total,
    human_reviewed:initial.human_reviewed,assistant_reviewed:initial.assistant_reviewed,
    evidence_images_loaded:initial.total*3,client_errors:errors,actual_review_writes:0,
    scope:'read_only_GUI_regression_not_label_approval'},null,2),{flag:'wx'});
  console.log(`Browser check passed: ${initial.total} examples, ${initial.total*3} native-evidence views, zero review writes.`);
}catch(e){await fs.writeFile(path.join(output,'failure.json'),JSON.stringify({error:String(e),errors,networkErrors},null,2),{flag:'wx'});throw e;}finally{ws.close();}
