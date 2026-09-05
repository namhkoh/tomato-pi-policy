@echo off
setlocal
cd /d "%~dp0\..\.."
call "D:\isaac-sim-6.0.1\python.bat" "%~dp0launch_sim_data.py" %*
exit /b %errorlevel%
