@echo off
setlocal
cd /d "%~dp0"
call run_full_robot_grasp_demo.cmd --scene package --sparse-contacts --finger-gravity --approach-tilt 10 --batch-gutter-visuals --local-wire-physics --context-gutters 3 --no-capture-milestones %*
exit /b %errorlevel%
