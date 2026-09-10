@echo off
setlocal
cd /d "%~dp0"
if defined ISAAC_SIM_PYTHON (
  call "%ISAAC_SIM_PYTHON%" -B -m sim_physics.benchmark %*
) else (
  call "D:\isaac-sim-6.0.1\python.bat" -B -m sim_physics.benchmark %*
)
exit /b %errorlevel%
