"use strict";
const $ = id => document.getElementById(id);
let state, current, saving = false, imageReady = false, viewed = new Set();
const sample = () => state?.samples.find(s => s.id === current);
function notice(message, error=false) { $("notice").textContent = message; $("notice").classList.toggle("error", error); }
function filtered() {
  const mode = $("filter").value;
  return (state?.samples || []).filter(s => mode === "all" || (mode === "pending" ? !s.decision :
    mode === "holds" ? !s.decision && s.suggestion?.suggested_decision === "hold" : Boolean(s.decision)))
    .slice().sort((a, b) => Number(b.suggestion?.suggested_decision === "hold") - Number(a.suggestion?.suggested_decision === "hold"));
}
function buttons() {
  const s = sample();
  const ready = Boolean(s && !s.decision && !saving && imageReady && viewed.has("rgb") && viewed.has("card") && $("inspected").checked && $("reviewer").value.trim() && $("notes").value.trim().length >= 20);
  const relation = $("suggestion-response").value;
  for (const b of document.querySelectorAll("[data-decision]")) {
    const d = b.dataset.decision;
    b.disabled = !ready || (d === "accept" && (s.source_blocks.length > 0 || (s.previous_decision && s.previous_decision.decision !== "accept"))) ||
      Boolean(s.suggestion && (!relation || (relation === "agree" && d !== s.suggestion.suggested_decision) || (relation === "disagree" && d === s.suggestion.suggested_decision)));
  }
  for (const id of ["filter", "sample", "prev", "next", "refresh"]) $(id).disabled = saving || (id !== "filter" && id !== "refresh" && !filtered().length);
  for (const b of document.querySelectorAll("[data-view]")) b.disabled = saving || !s;
  $("notes").disabled = saving || !s || Boolean(s.decision);
  $("inspected").disabled = saving || !s || Boolean(s.decision);
  $("copy-suggestion").disabled = saving || !s?.suggestion || Boolean(s.decision) || !viewed.has("rgb") || !viewed.has("card");
  $("suggestion-response").disabled = saving || !s?.suggestion || Boolean(s.decision);
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
  for (const s of list) { const o = document.createElement("option"); o.value = s.id; o.textContent = `${s.id} · ${s.difficulty}${s.suggestion?.suggested_decision === "hold" ? " · advisory HOLD" : ""}${s.decision ? " · " + s.decision.decision : ""}`; $("sample").append(o); }
  $("progress").textContent = `${state.pending} awaiting human review / ${state.total} · ${state.human_recorded} human decisions · ${state.suggestions_count} suggestions (${state.advisory_holds} advisory holds)`;
  $("records").textContent = "New decisions: " + state.records_directory + "; legacy assistant follow-ups: sibling human_decisions/";
  $("notes").value = ""; $("inspected").checked = false; viewed = new Set(); imageReady = false;
  $("suggestion-panel").hidden = true; $("suggestion-details").open = false; $("suggestion-response").value = "";
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
  $("prior-notes").textContent = s.previous_decision ? `Preserved assistant review: ${s.previous_decision.decision}. ${s.previous_decision.notes}` : s.decision?.notes || "No previous task review for this card.";
  $("notes").value = s.decision?.notes || "";
  const suggestion = s.suggestion;
  $("suggestion-panel").hidden = !suggestion;
  if (suggestion) {
    $("suggestion-title").textContent = `Assistant suggests: ${suggestion.suggested_decision.toUpperCase()} · not a human approval`;
    $("suggestion-notes").textContent = suggestion.notes;
    $("suggestion-source").textContent = `${suggestion.issue || "Visual assessment"} · ${suggestion.created_utc} · bound evidence ${suggestion.sha256.slice(0,12)}`;
    $("suggestion-response").value = s.decision?.suggestion_context?.response || "";
  }
  const previousHold = s.previous_decision && s.previous_decision.decision !== "accept";
  $("warning").hidden = !s.source_blocks.length && !previousHold;
  $("warning").textContent = "Source or earlier task hold/rejection is active. Accept cannot clear it. " + s.source_blocks.map(b => b.notes).join(" ") + (previousHold ? " " + s.previous_decision.notes : "");
  setView("rgb");
}
function move(delta) { if (saving) return; const list = filtered(), n = list.findIndex(s => s.id === current); if (list.length) render(list[(n + delta + list.length) % list.length].id); }
async function refresh() {
  if (saving) return;
  try {
    const response = await fetch("/api/state"); const data = await response.json(); if (!response.ok) throw new Error(data.error);
    if (data.gui_version !== "assistant_suggestions_and_human_followup.v1") {
      state = null; current = null; buttons();
      throw new Error("This server is running the older reviewer. Open the updated reviewer at http://127.0.0.1:8881, or restart the server with the updated code. No review was saved.");
    }
    state = data; render(current);
  }
  catch (e) { notice(e.message, true); }
}
async function save(decision) {
  const s = sample(); if (!s || saving || $(decision).disabled) return;
  saving = true; buttons(); notice("Verifying evidence and saving your decision…");
  try {
    const response = await fetch("/api/review", {method:"POST", headers:{"Content-Type":"application/json", "X-Review-Token":document.querySelector('meta[name="review-token"]').content},
      body:JSON.stringify({sample_id:s.id, reviewer:$("reviewer").value.trim(), decision, notes:$("notes").value.trim(), inspected:$("inspected").checked, bundle_sha256:state.bundle_sha256,
        suggestion_context:s.suggestion ? {sha256:s.suggestion.sha256, response:$("suggestion-response").value} : null})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Review was not saved");
    try { localStorage.setItem("greenhouse-v3-reviewer", $("reviewer").value.trim()); } catch (_) { /* Optional convenience. */ }
    state = data.state; saving = false;
    const next = filtered().find(x => !x.decision); render(next?.id || s.id);
    notice(`Saved ${s.id}: ${decision}. ${data.source_block_path ? "Source frame also blocked from export. " : ""}${next ? "Next pending card is shown." : `No pending cards remain in this filter; ${state.pending} still await human review in the bundle.`} This is not whole-dataset or physical-cut approval.`);
  } catch (e) { saving = false; notice(e.message + " Refresh to reconcile saved state before retrying.", true); buttons(); }
}
$("filter").onchange = () => render(); $("sample").onchange = () => render($("sample").value);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1); $("refresh").onclick = refresh;
for (const id of ["reviewer", "notes", "inspected", "suggestion-response"]) $(id).oninput = buttons;
$("copy-suggestion").onclick = () => {
  if ($("copy-suggestion").disabled) return;
  if ($("notes").value.trim() && !window.confirm("Replace your unsaved note with the assistant suggestion?")) return;
  $("notes").value = sample().suggestion.notes; $("notes").focus(); buttons();
  notice("Suggestion copied for editing. Nothing has been saved; your inspection and final decision are still required.");
};
for (const b of document.querySelectorAll("[data-view]")) b.onclick = () => setView(b.dataset.view);
for (const b of document.querySelectorAll("[data-decision]")) b.onclick = () => save(b.dataset.decision);
document.addEventListener("keydown", e => { if (["INPUT","TEXTAREA","SELECT","BUTTON"].includes(e.target.tagName) || e.ctrlKey || e.altKey || e.metaKey) return; if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); move(e.key === "ArrowLeft" ? -1 : 1); } });
try { $("reviewer").value = localStorage.getItem("greenhouse-v3-reviewer") || ""; } catch (_) { /* Optional convenience. */ }
refresh();
