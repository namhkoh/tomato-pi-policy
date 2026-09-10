@echo off
setlocal
cd /d "%~dp0"
if defined ISAAC_SIM_PYTHON (
  call "%ISAAC_SIM_PYTHON%" -B -m sim_data.active_perception_review --open %*
) else if exist "%USERPROFILE%\miniconda3\envs\egodelta_robot\python.exe" (
  "%USERPROFILE%\miniconda3\envs\egodelta_robot\python.exe" -B -m sim_data.active_perception_review --open %*
) else if exist "D:\isaac-sim-6.0.1\python.bat" (
  call "D:\isaac-sim-6.0.1\python.bat" -B -m sim_data.active_perception_review --open %*
) else (
  python -B -m sim_data.active_perception_review --open %*
)
if errorlevel 1 pause
endlocal
