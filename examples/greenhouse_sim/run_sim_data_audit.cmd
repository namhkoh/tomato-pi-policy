@echo off
setlocal
cd /d "%~dp0"
call "D:\isaac-sim-6.0.1\python.bat" -m sim_data.audit %*
exit /b %errorlevel%
