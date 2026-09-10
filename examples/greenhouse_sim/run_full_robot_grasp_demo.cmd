@echo off
setlocal
cd /d "%~dp0"
call run_physics_qualification.cmd --full-robot-probe --robot-interactive --gui --spring-mode implicit_effort --solver PGS --physics-hz 240 --physics-threads 1 --fabric --render-hz 30 --seconds 7 --force-newton 0 --grasp-arc-m .08 %*
exit /b %errorlevel%
