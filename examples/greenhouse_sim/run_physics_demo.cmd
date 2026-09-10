@echo off
setlocal
cd /d "%~dp0"
call run_physics_qualification.cmd --gui --interactive --scene isolated --spring-mode implicit_effort --solver PGS --physics-hz 240 --physics-threads 1 --fabric --render-hz 30 %*
exit /b %errorlevel%
