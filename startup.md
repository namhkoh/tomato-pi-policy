# Collaborator startup guide

Last checked: 2026-09-15. Repository: [namhkoh/tomato-pi-policy](https://github.com/namhkoh/tomato-pi-policy), branch **`koh-dev/sim-data`**. Setup checked against code revision `da175bc`; this guide does not imply that every local change or generated asset has been pushed.

## 1. What this repository does

We are developing tomato deleafing with an RB-Y1 robot: identify a safe petiole cut point, grasp when appropriate with the left arm, and cut with the right-hand knife. The longer-term system combines visual reasoning with geometry and feedback-controlled manipulation.

The immediate VLM task is narrower: **robot-head RGB plus a target query -> a cut-point prediction**, with visibility/abstention handling in the broader task. We are collecting a new, clearer dataset targeting **20,000 training images plus held-out sets**. That release is still being built; raw captures and automatic review candidates are not automatically approved training data.

There are three distinct workflows:

| Workflow | Entry point | Important limitation |
|---|---|---|
| Current supplied greenhouse + robot preview | `examples/greenhouse_sim/launch_sim_data.py` | Static preview/review; no grasping, cutting or dynamics. Start here. |
| Synthetic VLM data | `examples/greenhouse_sim/sim_data/` | Native RGB/depth, anatomy labels, review, export and evaluation. Collection needs a validated plan and assets, not just a viewport screenshot. |
| Physical grasp/cut experiments | `examples/greenhouse_sim/sim_physics/` | Separate experimental harness; individual passing cases do not establish general reliability or real-time performance. |

The root project is based on **OpenPI**. Installing its full training environment is **not necessary to view the greenhouse**. The older [greenhouse README](examples/greenhouse_sim/README.md) includes historical Isaac 5.1 and robot v1.0 instructions; use the current entry points below first.

## 2. Machine and Isaac Sim setup

The reproduced setup is **Windows + Isaac Sim 6.0.1**, using its bundled Python 3.12. An NVIDIA RTX-capable GPU, a compatible driver, and sufficient RAM, system commit headroom and SSD space are needed. The current development machine has an RTX 5090 with 32 GiB VRAM; this is a reference configuration, not a measured minimum requirement.

Follow NVIDIA's [workstation installation instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) and [hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html). Select **6.0.1** for reproduction; the documentation's `latest` version can be newer. Other versions require separate validation.

Example install directory: `D:\isaac-sim-6.0.1`. Confirm that `python.bat` exists. Run simulator scripts through that interpreter, **not Conda, the root OpenPI environment, or a separately installed `usd-core`**. Do not install the root `pyproject.toml` dependencies into Isaac's Python. In native scripts, initialize `SimulationApp` before importing `omni` or `pxr`.

Linux uses `python.sh` instead of `python.bat`, but the Windows launchers and local paths are not a verified Linux setup recipe. No physical robot, ROS installation, API key or model download is needed for the preview.

## 3. Get the code and assets

In PowerShell, from your preferred parent directory:

```powershell
git clone --branch koh-dev/sim-data --single-branch https://github.com/namhkoh/tomato-pi-policy.git
Set-Location tomato-pi-policy
git submodule update --init --recursive third_party/rby1-sdk
git rev-parse HEAD
git submodule status third_party/rby1-sdk
```

You need repository access. Keep the SDK at the revision pinned by the checkout; do not replace it with an arbitrary latest version. The current robot contract is **RB-Y1 Model A v1.2**, sourced from `third_party/rby1-sdk/models/rby1a/urdf/model_v1.2.urdf`.

**A Git clone does not include `data/`.** Ask the maintainer for:

- `tomato_greenhouse_pack.tar.gz`, the supplied greenhouse/plant archive, and its checksum.
- Any specific dataset/review package or recorded physics evidence you intend to reproduce. These are separate from the source checkout.

Extract the greenhouse archive into a **new, empty** directory; do not overwrite an existing source package. For the default layout:

```powershell
Get-FileHash C:\path\to\tomato_greenhouse_pack.tar.gz -Algorithm SHA256
tar -tzf C:\path\to\tomato_greenhouse_pack.tar.gz | Select-Object -First 12
New-Item -ItemType Directory -Path data\sim_data\package_20260905
tar -xzf C:\path\to\tomato_greenhouse_pack.tar.gz -C data\sim_data\package_20260905
Test-Path data\sim_data\package_20260905\tomato_greenhouse_pack\house\green_house_base.usd
```

Compare the hash with the maintainer's checksum. The final `Test-Path` must return `True`; if the archive has a different top-level layout, use `--package` to point to the directory containing `house`, `plants`, and `env_panel`.

The package includes 24 component-level plant families, 26 backdrop plants, greenhouse assets, and lighting-panel code. Its original plant generator is **not included**. Our generated variations retain their source-family ancestry; they are not independent biological donors. Preserve the package as read-only source data.

## 4. Build the robot once

Run all following commands from the repository root. Adjust only the Isaac installation path:

```powershell
$IsaacPython = 'D:\isaac-sim-6.0.1\python.bat'
$env:PYTHONPATH = "$PWD\examples;$PWD\examples\greenhouse_sim"
$env:OPENBLAS_NUM_THREADS = '1'
Test-Path $IsaacPython
Test-Path third_party\rby1-sdk\models\rby1a\urdf\model_v1.2.urdf
& $IsaacPython examples\greenhouse_sim\build_robot.py
Test-Path data\greenhouse_sim\robots\rby1a_v1.2\rby1a_v1.2.usd
```

The builder uses the tracked CAD/mesh exports and mount manifest under `greenhouse/robot_assets/`, including `derived/hardware_v1.2.json`. FreeCAD is not needed just to use these existing exports. Build locally after relocating a checkout: generated USD/configuration layers can reference machine-specific asset paths. If sharing a prebuilt robot, include its complete directory and dependencies, not only the top-level USD.

The base asset contains the fitted knife and cameras. The **static preview defaults to restoring the right gripper**; its session-only tool selector can instead show the knife. This does not enable cutting physics.

## 5. Launch the current greenhouse

Use a fresh output directory for each run:

```powershell
$PreviewRun = "data\sim_data\collaborator_preview_$(Get-Date -Format yyyyMMdd_HHmmss)"
& $IsaacPython examples\greenhouse_sim\launch_sim_data.py --output $PreviewRun
```

For a non-default package location, append:

```powershell
--package 'D:\your-assets\tomato_greenhouse_pack'
```

The last block is an argument to the launch command, not a separate command. With the default installation path, `examples\greenhouse_sim\run_sim_data.cmd --output $PreviewRun` is a convenience wrapper; it hardcodes `D:\isaac-sim-6.0.1\python.bat`.

Expect the supplied `house/green_house_base.usd`, three populated gutters, two component-level foreground plants, backdrop instances, and the RB-Y1. The default view is **Robot head D405**. The **Tomato package preview** panel switches between head, wrist and inspection views. Preview resolution is **848x408**. Unbundled external prop payloads are explicitly excluded; this is logged. Source USDs are not rewritten.

Useful optional arguments:

- `--right-tool knife_only`: show the fitted right-hand knife instead of the original gripper.
- `--review`: open the anatomy-review workflow; this can save review decisions, so use a separate output and coordinate with the dataset owner.
- `--no-robot`: diagnose greenhouse assets without a robot build.
- `--headless --frames 120`: bounded preview smoke test without an interactive window.

Wait for **`PACKAGE_READY`** in the console. The output directory contains `status.json` and `preview.png`. In a second PowerShell window, inspect the actual run path:

```powershell
Get-Content data\sim_data\YOUR_PREVIEW_RUN\status.json -Raw
```

An interactive ready preview reports `state: running`; a bounded run finishes as `completed`. A process exit code of zero alone is not proof of success. The preview does not respond with physical plant motion when you press Play: dynamics are intentionally disabled here.

## 6. Where to work next

| Path | Purpose |
|---|---|
| `examples/greenhouse_sim/sim_data/` | Anatomy, native capture, target-query labels, review GUIs, release validation, Qwen adapters. |
| `examples/greenhouse_sim/sim_data/native_capture_v2/` | Opt-in wider actual robot-head viewpoint search and native collection. |
| `examples/greenhouse_sim/sim_data/native_dataset/` | Lossless native storage, annotation replay and candidate accounting; not final release approval. |
| `examples/greenhouse_sim/sim_physics/` | Current-package physical grasp/cut qualification. Read its README before launching. |
| `examples/greenhouse_sim/greenhouse_sim/` | Shared robot model, FK/IK, mounts, plus substantial legacy scene/physics/RL code. |
| `examples/greenhouse_sim/vlm_eval/` | Endpoint-based VLM evaluation and visual grounding experiments. |
| `src/openpi/`, `packages/openpi-client/` | Upstream-derived policy training/inference and client infrastructure. |
| `data/sim_data/` | Local-only packages, captures, generated plants, review receipts and training archives. Not included in Git. |
| `data/sim_physics/` | Local-only physics reports, logs and recordings. |

Read [vlm_train_data.md](vlm_train_data.md) for the current data plan and dated progress, and [dev.md](dev.md) for engineering history. Both contain historical checkpoints; use their latest relevant entries rather than treating every earlier result as current.

The [native dataset README](examples/greenhouse_sim/sim_data/native_dataset/README.md) explains the new **1696x816 native** capture path. These images use the same mounted-camera geometry, not upscaled 848x408 images; this is a synthetic capture mode, not a claim about physical D405 resolution. Depth is Isaac's native optical-axis Z in metres; a heatmap is only a visualization. Masks and hidden geometry are supervision/evaluation sidecars, not automatically VLM inputs.

Review GUIs are local browser tools, separate from Isaac. For example, after receiving the compatible audit and its referenced data, `python -m sim_data.review_gui --audit PATH_TO_AUDIT_JSON --port 8877 --open` works with the `PYTHONPATH` above and a CPU Python containing NumPy/Pillow. Do not use historical default audit paths on a fresh clone, or share an audit JSON without its referenced images. Different review stages use different schemas/tools.

For physics, see [sim_physics/README.md](examples/greenhouse_sim/sim_physics/README.md). `run_neutral_cut_demo.cmd` additionally depends on local `native293/report.json` evidence that Git does not contain. The static preview is not a substitute for that harness. Legacy teleop/PPO code is not enabled by this startup; do not connect to or command hardware as part of onboarding.

For server-side VLM work, read [H200_RESULTS.md](examples/greenhouse_sim/sim_data/H200_RESULTS.md) and [H200_RUNBOOK.md](examples/greenhouse_sim/sim_data/H200_RUNBOOK.md). The runbook describes the earlier visible/occluded 848x408 release; it is **not** the finalized recipe for the pending 20k high-resolution clear-point dataset. Match dataset manifest, coordinate contract, adapter and code revision. No model weights or training are required on the simulator PC.

## 7. Checks and common problems

- **Missing robot or textures:** initialize the pinned SDK, verify the source package tree, and rebuild the Model A v1.2 asset. Do not fall back silently to a v1.0 asset.
- **`pxr`/`omni` import errors:** use Isaac's bundled interpreter and correct startup import order; do not mix pip USD binaries with Kit.
- **Slow or unresponsive startup:** run one Isaac process at a time and let assets/shaders load. Check console and JSON reports. Current capture scheduling reserves 20 GiB of Windows commit headroom; this is a launch guard, not a RAM minimum or a runtime guarantee. Do not disable memory guards or silently remove greenhouse geometry to make a run pass.
- **Review port busy:** select another unused loopback port rather than stopping someone else's review process.
- **Capture job looks successful:** require its completion report, native checks and absence of `failure.json`; saved RGB alone is not an accepted record.
- **Changes invalidate a capture plan:** plans bind source/configuration/code hashes. Use a fresh plan/run; do not edit receipts, original captures or frozen splits to bypass validation.

A small CPU-only accounting regression, using a separate environment with `pytest` installed:

```powershell
$env:PYTHONPATH = "$PWD\examples;$PWD\examples\greenhouse_sim"
python -m pytest -q examples/greenhouse_sim/sim_data/native_dataset/test_admission.py
```

This test needs neither Isaac nor greenhouse assets and does not validate rendering or physics. Before handing work back, record `git rev-parse HEAD`, the SDK revision and run/report paths. Never commit API keys, model weights, caches or the entire `data/` directory.
