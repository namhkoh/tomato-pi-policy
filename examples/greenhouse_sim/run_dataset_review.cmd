@echo off
setlocal
cd /d "%~dp0"
if defined ISAAC_SIM_PYTHON (
  call "%ISAAC_SIM_PYTHON%" -m sim_data.review_gui --open %*
) else if exist "D:\isaac-sim-6.0.1\python.bat" (
  call "D:\isaac-sim-6.0.1\python.bat" -m sim_data.review_gui --open %*
) else (
  python -m sim_data.review_gui --open %*
)
if errorlevel 1 pause
endlocal
