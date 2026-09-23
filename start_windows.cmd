@echo off
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
pushd "%~dp0"
if not exist "logs" mkdir "logs"

if exist "venv\Scripts\python.exe" goto run

where py >nul 2>>"logs\windows_service.log"
if errorlevel 1 goto no_python
py -3 -m venv venv >>"logs\windows_service.log" 2>&1
if errorlevel 1 goto failed
"venv\Scripts\python.exe" -m pip install --upgrade pip setuptools >>"logs\windows_service.log" 2>&1
if errorlevel 1 goto failed
"venv\Scripts\python.exe" -m pip install -r requirements.txt >>"logs\windows_service.log" 2>&1
if errorlevel 1 goto failed

:run
"venv\Scripts\python.exe" main.py --ai-engine pikafish %* >>"logs\windows_service.log" 2>&1
if errorlevel 1 goto failed
popd
exit /b 0

:no_python
echo Python was not found. Install Python 3.11 or newer and try again. >>"logs\windows_service.log"
goto failed_pause

:failed
echo The synchronizer could not start. >>"logs\windows_service.log"

:failed_pause
popd
exit /b 1
