@echo off
setlocal

set "WORLDBANK_ROOT=%~dp0..\.."
if not defined WORLDBANK_ISAAC_PYTHON set "WORLDBANK_ISAAC_PYTHON=D:\isaac-sim\python.bat"
set "WORLDBANK_RECORD_ARGS="
if /I "%~1"=="record" set "WORLDBANK_RECORD_ARGS=--teleop-record-dir data\greenhouse_sim\worldbank_demonstrations"

if not exist "%WORLDBANK_ISAAC_PYTHON%" (
    echo Isaac Sim Python launcher not found: %WORLDBANK_ISAAC_PYTHON%
    exit /b 1
)

pushd "%WORLDBANK_ROOT%"
"%WORLDBANK_ISAAC_PYTHON%" examples\greenhouse_sim\interactive_greenhouse.py ^
    --scene data\greenhouse_sim\scenes\deleafing_bench.usd ^
    --vine-dir greenhouse\tomato_glb_20 ^
    --robot data\greenhouse_sim\robots\rby1a_v1.0.usd ^
    --physics-vines 1 ^
    --target-vine Vine_0002 ^
    --target-organ SubStem_00 ^
    --robot-position-mode fixed ^
    --robot-position 10.639221515539253 4.25 -0.15254085567917297 ^
    --teleop-command-file data\greenhouse_sim\teleop_command.json ^
    --teleop-contact-policy monitor ^
    --contact-diagnostics ^
    --report data\greenhouse_sim\worldbank_demo_live.json ^
    %WORLDBANK_RECORD_ARGS%
set "WORLDBANK_EXIT=%ERRORLEVEL%"
popd
exit /b %WORLDBANK_EXIT%
