"use strict";
const $ = id => document.getElementById(id);
let state = null, current = null, view = "card", saving = false, imageReady = false;
const captions = {
  card: "All evidence: full-scene robot-head RGB above, labelled review-only crops below. Click to open full size.",
  rgb: "Original 848×408 robot-head RGB. No annotation, crop or cinematic camera in this model input.",
  overlay: "Projected prototype labels on the robot view. The overlay alone does not establish visibility or cut safety.",
  mask: "Green highlights exactly the visible petiole pixels from native renderer identity. No guessed or amodal mask.",
  depth: "Direct saved Isaac camera-Z depth, in metres. Yellow nearer, purple farther; grey is invalid. No RGB-estimated depth."
};
function notice(text, error=false) { $("notice").textContent = text; $("notice").classList.toggle("error", error); }
function filtered() {
  if (!state) return [];
  return state.samples.filter(s => $("filter").value === "all" || ($("filter").value === "pending" ? !s.human_review : s.recommended));
}
function sample() { return state?.samples.find(s => s.sample_id === current); }
function updateButtons() {
  const s = sample(), ready = Boolean(s && imageReady && !saving && $("reviewer").value.trim() && $("inspected").checked);
  for (const button of document.querySelectorAll("[data-decision]")) button.disabled = !ready || (button.dataset.decision === "confirm" && !s.can_confirm);
  for (const id of ["filter", "sample", "prev", "next"]) $(id).disabled = saving || !filtered().length;
  // The filter must remain usable when the current set becomes empty.
  $("filter").disabled = saving;
}
function setView(kind) {
  view = kind;
  for (const button of document.querySelectorAll("[data-view]")) {
    const selected = button.dataset.view === view;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  const s = sample();
  if (!s) return;
  imageReady = false;
  const expected = s.images[view];
  $("evidence").onload = () => { if ($("evidence").getAttribute("src") === expected) { imageReady = true; updateButtons(); } };
  $("evidence").onerror = () => { imageReady = false; updateButtons(); notice("Evidence failed to load. Refresh before reviewing.", true); };
  $("evidence").src = expected;
  $("full-image").href = expected;
  $("caption").textContent = captions[view];
  updateButtons();
}
function render(preferred=null) {
  const list = filtered();
  current = list.some(s => s.sample_id === preferred) ? preferred : (list.find(s => !s.human_review) || list[0])?.sample_id;
  const select = $("sample"); select.replaceChildren();
  for (const s of list) {
    const option = document.createElement("option");
    option.value = s.sample_id;
    option.textContent = `${s.sample_id.replace("sample_", "")} · ${s.target}${s.human_review ? " · " + s.human_review.decision : " · pending"}`;
    select.append(option);
  }
  const recommended = state.samples.filter(s => s.recommended);
  $("run-name").textContent = "Capture: " + (state.capture_run_name || "original reviewed pilot");
  $("filter").options[0].textContent = `Recommended (${recommended.length})`;
  $("progress").textContent = `${recommended.filter(s => s.human_review).length}/${recommended.length} recommended reviewed · ${state.reviewed}/${state.total} total`;
  $("records").textContent = "Saved to: " + state.records_directory;
  $("inspected").checked = false;
  $("notes").value = "";
  const s = sample();
  if (!s) {
    $("target").textContent = "This review set is complete";
    $("saved-status").textContent = "Choose All samples to revisit a decision";
    $("evidence").removeAttribute("src"); $("full-image").removeAttribute("href");
    for (const id of ["caption", "assistant", "metrics", "camera"]) $(id).textContent = "";
    $("gate-warning").hidden = true;
    updateButtons(); return;
  }
  select.value = current;
  $("target").textContent = `${s.target} / ${s.sample_id.replace("sample_", "")}`;
  $("saved-status").textContent = s.human_review ? `Saved: ${s.human_review.decision} · ${s.human_review.reviewer}` : "No human decision yet";
  if (s.human_review) $("notes").value = s.human_review.notes;
  $("assistant").textContent = s.assistant_review?.notes || "Numerical audit complete. No explicit assistant visual-review decision for this sample.";
  $("metrics").textContent = `Petiole width: ${s.quality.estimated_petiole_diameter_px.toFixed(2)} px · interval: ${s.quality.projected_interval_length_px.toFixed(2)} px · sampled interval visibility: ${Math.round(100*s.visibility)}%. These are provisional engineering checks.`;
  $("camera").textContent = `Mounted robot POV verified: ${s.camera.mounted_robot_pov_verified ? "yes" : "no"}. ${s.camera.camera_path}`;
  $("camera").classList.add("path");
  let warning = "";
  if (!s.can_confirm) warning = "Confirm is disabled: this sample failed the numerical gate. " + s.quality.clear_view_rejection_reasons.join(", ") + ". Hold or reject instead.";
  else if (["hold", "reject"].includes(s.assistant_review?.decision)) warning = "An engineering hold/rejection remains. Your confirmation can record visual agreement, but cannot clear that separate issue or approve training.";
  $("gate-warning").textContent = warning; $("gate-warning").hidden = !warning;
  $("help").textContent = s.human_review ? "Saving again revises this human decision and preserves its history." : "Enter your name, inspect the evidence and tick the checkbox to save.";
  setView("card");
}
function move(delta) {
  if (saving) return;
  const list = filtered(), i = list.findIndex(s => s.sample_id === current);
  if (list.length) render(list[(i + delta + list.length) % list.length].sample_id);
}
async function save(decision) {
  const s = sample();
  if (!s || saving || $(decision).disabled) return;
  saving = true; updateButtons(); notice("Checking source integrity and saving your review…");
  try {
    const response = await fetch("/api/review", {method: "POST", headers: {"Content-Type": "application/json", "X-Review-Token": document.querySelector('meta[name="review-token"]').content},
      body: JSON.stringify({sample_id: s.sample_id, reviewer: $("reviewer").value.trim(), decision,
        notes: $("notes").value, inspected: $("inspected").checked, expected_review_id: s.human_review?.review_id || null})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Review was not saved");
    state = result.state;
    try { localStorage.setItem("greenhouse-reviewer", $("reviewer").value.trim()); } catch (_) { /* Optional convenience only. */ }
    const next = filtered().find(item => !item.human_review);
    saving = false; render(next?.sample_id || s.sample_id);
    notice(`Saved ${s.sample_id}: ${decision}. ${next ? "Next pending sample is shown." : "This set is reviewed; you can revisit it or choose All samples."} No training or physical-cut approval was granted.`);
  } catch (error) {
    saving = false; notice(error.message, true); updateButtons();
  }
}
$("filter").onchange = () => render();
$("sample").onchange = () => render($("sample").value);
$("prev").onclick = () => move(-1); $("next").onclick = () => move(1);
$("reviewer").oninput = updateButtons; $("inspected").onchange = updateButtons;
for (const button of document.querySelectorAll("[data-view]")) button.onclick = () => { if (!saving) setView(button.dataset.view); };
for (const button of document.querySelectorAll("[data-decision]")) button.onclick = () => save(button.dataset.decision);
document.addEventListener("keydown", event => {
  if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(event.target.tagName) || event.ctrlKey || event.altKey || event.metaKey) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); move(event.key === "ArrowLeft" ? -1 : 1); }
});
async function init() {
  try {
    try { $("reviewer").value = localStorage.getItem("greenhouse-reviewer") || ""; } catch (_) { /* Storage may be disabled. */ }
    const response = await fetch("/api/state"); const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Failed to load review set");
    state = result; render();
  } catch (error) { notice(error.message, true); }
}
init();
