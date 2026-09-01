@echo off
setlocal

set "WORLDBANK_ROOT=%~dp0..\.."
if not defined WORLDBANK_RBY1_PYTHON set "WORLDBANK_RBY1_PYTHON=C:\Users\USER\miniconda3\envs\egodelta_robot\python.exe"
if not defined WORLDBANK_RBY1_ADDRESS set "WORLDBANK_RBY1_ADDRESS=192.168.12.1:50051"

if not exist "%WORLDBANK_RBY1_PYTHON%" (
    echo RB-Y1 Python environment not found: %WORLDBANK_RBY1_PYTHON%
    exit /b 1
)

pushd "%WORLDBANK_ROOT%"
"%WORLDBANK_RBY1_PYTHON%" examples\greenhouse_sim\rby1_robot_state_to_sim.py ^
    --address "%WORLDBANK_RBY1_ADDRESS%" ^
    --command-file data\greenhouse_sim\teleop_command.json
set "WORLDBANK_EXIT=%ERRORLEVEL%"
popd
exit /b %WORLDBANK_EXIT%
