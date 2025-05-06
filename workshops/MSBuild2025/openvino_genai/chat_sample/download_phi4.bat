@echo off
setlocal enabledelayedexpansion

:: Check if virtual environment exists
if not exist "..\venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

:: Run the application
echo Download Phi 4 from huggingface..
huggingface-cli download OpenVINO/Phi-4-mini-instruct-int4-ov --local-dir Phi-4-mini-instruct-int4-ov

:: Keep console open after execution
pause
exit