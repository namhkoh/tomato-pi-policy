// Read-only browser regression. An isolated headless Chromium CDP must already
// listen on 127.0.0.1:9223. Never submits a review on real dataset cards.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const [base = 'http://127.0.0.1:8881', output, pilot] = process.argv.slice(2);
assert(output, 'Supply a new screenshot/report directory');
const url = new URL(base);
assert.equal(url.hostname, '127.0.0.1');
await fs.mkdir(output); // Never overwrite previous browser evidence.
const initial = await fetch(base+'/api/state').then(r => r.json());
assert.equal(initial.gui_version, 'assistant_suggestions_and_human_followup.v1');
const targets = await fetch('http://127.0.0.1:9223/json').then(r => r.json());
const target = targets.find(t => t.type === 'page');
assert(target);
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen=resolve; ws.onerror=reject; });
let seq=0; const pending=new Map(), errors=[];
ws.onmessage = event => {
  const message=JSON.parse(event.data);
  if (message.id) {
    const p=pending.get(message.id); if (!p) return;
    pending.delete(message.id); clearTimeout(p.timer);
    if (message.error) p.reject(new Error(JSON.stringify(message.error))); else p.resolve(message.result);
  } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
};
function cdp(method, params={}) {
  return new Promise((resolve, reject) => {
    const id=++seq;
    const timer=setTimeout(() => { pending.delete(id); reject(new Error('CDP timeout: '+method)); }, 15000);
    pending.set(id,{resolve,reject,timer}); ws.send(JSON.stringify({id,method,params}));
  });
}
async function evaluate(expression) {
  const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  assert(!r.exceptionDetails, JSON.stringify(r.exceptionDetails)); return r.result.value;
}
async function until(expression) {
  for(let i=0;i<80;i++) {
    if(await evaluate(expression)) return;
    await new Promise(r => setTimeout(r,100));
  }
  throw new Error('Browser condition timed out: '+expression);
}
async function screenshot(name) {
  const r=await cdp('Page.captureScreenshot',{format:'png'});
  await fs.writeFile(path.join(output,name),Buffer.from(r.data,'base64'),{flag:'wx'});
}
try {
  await cdp('Page.enable'); await cdp('Runtime.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1450,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:base});
  await until('document.querySelector("#evidence")?.naturalWidth === 848 && document.querySelector("#sample")?.options.length > 0');
  assert(await evaluate('document.querySelector("#accept").disabled && !document.querySelector("#inspected").checked && document.querySelector("#notes").value === ""'));
  assert(await evaluate('!document.querySelector("#suggestion-panel").hidden'));
  await evaluate('document.querySelector("#suggestion-details").open = true');
  await screenshot('original_with_suggestion.png');
  await evaluate('document.querySelector("[data-view=card]").click()');
  await until('document.querySelector("#evidence").naturalWidth === 1152');
  assert(await evaluate('document.querySelector("#accept").disabled'));
  // Exercise editable advice, but DO NOT enter a human name, inspect checkbox or save.
  await evaluate('document.querySelector("#copy-suggestion").click()');
  assert(await evaluate('document.querySelector("#notes").value.length >= 20 && document.querySelector("#accept").disabled'));
  await screenshot('evidence_with_editable_advice.png');
  await evaluate('document.querySelector("#notes").value = ""; document.querySelector("#filter").value="holds"; document.querySelector("#filter").dispatchEvent(new Event("change"))');
  const holdCount = await evaluate('document.querySelector("#sample").options.length');
  assert(holdCount <= initial.advisory_holds);
  // A valid image source always starts with the unmarked RGB, even after filters.
  if(holdCount) await until('document.querySelector("#evidence").naturalWidth === 848');
  if(pilot) {
    const manifest=JSON.parse(await fs.readFile(pilot,'utf8'));
    await cdp('Page.navigate',{url:pathToFileURL(path.join(path.dirname(pilot),'review.html')).href});
    await until('document.querySelectorAll("section").length > 0 && Array.from(document.images).every(i => i.complete && i.naturalWidth === 848)');
    assert.equal(await evaluate('document.querySelectorAll("section").length'),manifest.examples.length);
    await evaluate('document.querySelector("#markers").checked = true');
    for(let i=0;i<manifest.examples.length;i++) {
      if(manifest.examples[i].kind !== 'invalid_candidate') continue;
      await evaluate(`document.querySelectorAll('section')[${i}].scrollIntoView({block:'start'})`);
      await screenshot(`negative_${i}_${manifest.examples[i].source_plant_family}.png`);
    }
  }
  const final=await fetch(base+'/api/state').then(r=>r.json());
  assert.deepEqual(final.samples.map(s=>s.decision),initial.samples.map(s=>s.decision),'Review records changed during read-only check');
  assert.equal(errors.length,0,JSON.stringify(errors));
  await fs.writeFile(path.join(output,'browser_check.json'),JSON.stringify({checked_utc:new Date().toISOString(),
    gui_version:initial.gui_version,total:initial.total,human_recorded:initial.human_recorded,
    suggestions:initial.suggestions_count,advisory_holds:initial.advisory_holds,hold_filter_count:holdCount,
    client_errors:errors,actual_review_writes:0,scope:'read_only_GUI_regression_not_label_approval'},null,2),{flag:'wx'});
  console.log('Read-only browser check passed:',output);
} finally { ws.close(); }
