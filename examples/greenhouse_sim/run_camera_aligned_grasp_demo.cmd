@echo off
setlocal
cd /d "%~dp0"
echo Grasp-only inspection: closer exact grasp and camera-aligned knife.
echo Native hold verified; the new downward cut is NOT yet qualified.
echo Requires a NEW --output directory. Press Run in the GUI when ready.
call run_physics_qualification.cmd --scene package --full-robot-probe --bimanual-cut --bimanual-hold-control --solver PGS --spring-mode implicit_effort --force-newton 0 --physics-hz 240 --physics-threads 4 --fabric --local-wire-physics --context-gutters 3 --batch-gutter-visuals --sparse-contacts --finger-gravity --compliant-fingers --finger-actuator-limit-n .8 --solve-articulation-contact-last --cut-model signed_edge_load_brittle_seam_v1 --knife-alignment camera --cut-style downward --grasp-arc-m .060 --exact-grasp-arc --grasp-roll 180 --grasp-depth-m .125 --grasp-compression-m .0005 --approach-tilt 10 --approach-distance .02 --station-pose .530398411918 .553263301559 -163.12111184 --seconds 20 --gui --robot-interactive --no-robot-auto-run --render-hz 30 --no-capture-milestones --no-physics-profiler %*
exit /b %errorlevel%
