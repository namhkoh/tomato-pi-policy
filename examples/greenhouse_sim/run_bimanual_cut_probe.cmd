@echo off
setlocal
cd /d "%~dp0"
echo EXPERIMENTAL qualification: not a verified grasp-cut-retain demonstration.
echo Requires a NEW --output directory. Failed guards never force a cut.
call run_physics_qualification.cmd --full-robot-probe --bimanual-cut --spring-mode implicit_effort --solver PGS --physics-hz 240 --physics-threads 1 --fabric --render-hz 30 --seconds 20 --force-newton 0 --grasp-arc-m .08 --scene package --sparse-contacts --finger-gravity --approach-tilt 10 --batch-gutter-visuals --local-wire-physics --context-gutters 3 %*
exit /b %errorlevel%
