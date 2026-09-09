"use strict";
const $ = id => document.getElementById(id);
let state, current, saving = false, imageReady = false, viewed = new Set();
const sample = () => state?.samples.find(s => s.id === current);
function notice(message, error=false) { $("notice").textContent = message; $("notice").classList.toggle("error", error); }
function filtered() { return (state?.samples || []).filter(s => $("filter").value === "all" || ($("filter").value === "pending" ? !s.decision : Boolean(s.decision))); }
function buttons() {
  const s = sample();
  const ready = Boolean(s && !s.decision && !saving && imageReady && viewed.has("rgb") && viewed.has("card") && $("inspected").checked && $("reviewer").value.trim() && $("notes").value.trim().length >= 20);
  for (const b of document.querySelectorAll("[data-decision]")) b.disabled = !ready || (b.dataset.decision === "accept" && s.source_blocks.length > 0);
  for (const id of ["filter", "sample", "prev", "next", "refresh"]) $(id).disabled = saving || (id !== "filter" && id !== "refresh" && !filtered().length);
  for (const b of document.querySelectorAll("[data-view]")) b.disabled = saving || !s;
  $("notes").disabled = saving || !s || Boolean(s.decision);
  $("inspected").disabled = saving || !s || Boolean(s.decision);
  $("help").textContent = s?.decision ? "Recorded decisions are read-only; no earlier review will be overwritten." : "Open both image views, enter your name and a short note, then tick the inspection checkbox.";
}
function setView(kind) {
  const s = sample(); if (!s || saving) return;
  imageReady = false;
  const expected = s.images[kind], sid = s.id;
  for (const b of document.querySelectorAll("[data-view]")) { const on = b.dataset.view === kind; b.classList.toggle("selected", on); b.setAttribute("aria-pressed", String(on)); }
  $("evidence").onload = () => { if (current === sid && $("evidence").getAttribute("src") === expected) { imageReady = true; viewed.add(kind); buttons(); } };
  $("evidence").onerror = () => { imageReady = false; buttons(); notice("Image failed integrity/loading checks. Refresh before reviewing.", true); };
  $("evidence").src = expected; $("full-image").href = expected;
  $("caption").textContent = kind === "rgb" ? "Untouched 848×408 robot-head RGB. First judge whether the petiole and junction are distinguishable without a mask." : "Saved evidence: full scene above, nominal cut and native mask/depth crops in the middle, separate target-query crops at the bottom. Hidden cuts have no white/magenta marks.";
  buttons();
}
function render(preferred=null) {
  const list = filtered();
  current = list.some(s => s.id === preferred) ? preferred : list[0]?.id;
  $("sample").replaceChildren();
  for (const s of list) { const o = document.createElement("option"); o.value = s.id; o.textContent = `${s.id} · ${s.difficulty}${s.decision ? " · " + s.decision.decision : ""}`; $("sample").append(o); }
  $("progress").textContent = `${state.pending} pending / ${state.total} cards · ${state.human_recorded} human decisions`;
  $("records").textContent = "Saved to: " + state.records_directory;
  $("notes").value = ""; $("inspected").checked = false; viewed = new Set(); imageReady = false;
  const s = sample();
  if (!s) {
    $("target").textContent = "No cards in this filter";
    for (const id of ["sample-id", "saved-status", "task", "answer", "caption", "prior-notes"]) $(id).textContent = "";
    $("evidence").removeAttribute("src"); $("full-image").removeAttribute("href"); $("warning").hidden = true; buttons(); return;
  }
  $("sample").value = current; $("target").textContent = s.target;
  $("sample-id").textContent = s.id + " · " + s.split + " / " + s.difficulty;
  $("saved-status").textContent = s.decision ? `${s.decision.decision} · ${s.decision.reviewer_role}: ${s.decision.reviewer}` : "Pending · no decision recorded";
  $("task").textContent = `Input query [u, v]: [${s.query.join(", ")}] — check this target, then assess the answer below.`;
  $("answer").textContent = JSON.stringify(s.answer, null, 2);
  $("prior-notes").textContent = s.decision?.notes || "No previous task review for this card.";
  $("notes").value = s.decision?.notes || "";
  $("warning").hidden = !s.source_blocks.length;
  $("warning").textContent = "Source hold/rejection is active. Accept cannot clear it. " + s.source_blocks.map(b => b.notes).join(" ");
  setView("rgb");
}
function move(delta) { if (saving) return; const list = filtered(), n = list.findIndex(s => s.id === current); if (list.length) render(list[(n + delta + list.length) % list.length].id); }
async function refresh() {
  if (saving) return;
  try { const response = await fetch("/api/state"); const data = await response.json(); if (!response.ok) throw new Error(data.error); state = data; render(current); }
  catch (e) { notice(e.message, true); }
}
async function save(decision) {
  const s = sample(); if (!s || saving || $(decision).disabled) return;
  saving = true; buttons(); notice("Verifying evidence and saving your decision…");
  try {
    const response = await fetch("/api/review", {method:"POST", headers:{"Content-Type":"application/json", "X-Review-Token":document.querySelector('meta[name="review-token"]').content},
      body:JSON.stringify({sample_id:s.id, reviewer:$("reviewer").value.trim(), decision, notes:$("notes").value.trim(), inspected:$("inspected").checked, bundle_sha256:state.bundle_sha256})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Review was not saved");
    try { localStorage.setItem("greenhouse-v3-reviewer", $("reviewer").value.trim()); } catch (_) { /* Optional convenience. */ }
    state = data.state; saving = false;
    const next = state.samples.find(x => !x.decision); render(next?.id || s.id);
    notice(`Saved ${s.id}: ${decision}. ${data.source_block_path ? "Source frame also blocked from export. " : ""}${next ? "Next pending card is shown." : "No pending cards remain in this bundle."} This is not whole-dataset or physical-cut approval.`);
  } catch (e) { saving = false; notice(e.message + " Refresh to reconcile saved state before retrying.", true); buttons(); }
}
$("filter").onchange = () => render(); $("sample").onchange = () => render($("sample").value);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1); $("refresh").onclick = refresh;
for (const id of ["reviewer", "notes", "inspected"]) $(id).oninput = buttons;
for (const b of document.querySelectorAll("[data-view]")) b.onclick = () => setView(b.dataset.view);
for (const b of document.querySelectorAll("[data-decision]")) b.onclick = () => save(b.dataset.decision);
document.addEventListener("keydown", e => { if (["INPUT","TEXTAREA","SELECT","BUTTON"].includes(e.target.tagName) || e.ctrlKey || e.altKey || e.metaKey) return; if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); move(e.key === "ArrowLeft" ? -1 : 1); } });
try { $("reviewer").value = localStorage.getItem("greenhouse-v3-reviewer") || ""; } catch (_) { /* Optional convenience. */ }
refresh();
