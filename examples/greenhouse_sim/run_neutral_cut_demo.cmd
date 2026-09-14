@echo off
setlocal
if /i "%~1"=="bimanual" goto bimanual
if /i "%~1"=="right_only" goto direct
goto usage
:bimanual
set "CUT_MODE=bimanual"
set "CUT_OPTIONS=--coupled-fingers-trial --approach-distance .08 --retained-separation-trial"
goto launch
:direct
set "CUT_MODE=right_only"
set "CUT_OPTIONS=--park-left-ready"
:launch
if "%~2"=="" goto usage
if not "%~3"=="" goto usage
pushd "%~dp0..\.." || exit /b 1
if not exist "D:\isaac-sim-6.0.1\python.bat" goto missing
if not exist "data\sim_physics\bimanual_downward_20260914_native293\report.json" goto missing
set "PYTHONPATH=%CD%\examples\greenhouse_sim;%CD%\examples"
set "OPENBLAS_NUM_THREADS=1"
echo Isolated neutral-start fixture. Press Run once after the panel is ready.
echo No hardware commands. Not greenhouse qualification or calibrated tissue cutting.
call D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial ^
 --output "%~2" --mode %CUT_MODE% --milestone cut_action --watch --capture ^
 --process-zone-trial --through-stroke-trial --material-clearance-trial --postcut-egress-trial ^
 --blade-aim-offset-m .0015 --rectilinear-floor-contacts --physics-threads 1 ^
 --postrelease-feed-m-s .001 --support-aware-feed-trial --neutral-ready-start ^
 --joint-transit-fallback --budgeted-joint-gravity --torso-degrees 0 0 0 0 0 0 ^
 --right-ready-degrees 0 -5 0 -120 0 70 0 ^
 --cut-priority-report data/sim_physics/bimanual_downward_20260914_native293/report.json ^
 --station-pose .52 .75 -147.10477763841965 ^
 --left-ik-seed-degrees -61.441133410536075 34.73085008787647 -40.46592774778478 -59.03111400330014 -25.317059074857838 -66.93979217019664 -70.29296927074824 ^
 --right-entry-seed-degrees -80.13688562959481 -49.431940487107774 91.29251949519043 -42.73954525420369 -23.393606797739935 -80.1724133309855 15.034122647754703 ^
 %CUT_OPTIONS%
set "CUT_EXIT=%ERRORLEVEL%"
popd
exit /b %CUT_EXIT%
:missing
echo Missing Isaac 6.0.1 or the local native293 cut-family evidence. See sim_physics/README.md.
popd
exit /b 2
:usage
echo Usage: %~nx0 bimanual^|right_only NEW_OUTPUT_DIRECTORY
echo Use a new directory for each one-shot run. There is no reset/replay or hardware control.
exit /b 2
