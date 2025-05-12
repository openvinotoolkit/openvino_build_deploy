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
echo Download and Compress Phi 3 from huggingface..
optimum-cli export openvino -m microsoft/Phi-3-mini-4k-instruct  --trust-remote-code --weight-format int4 --sym --ratio 1.0 --group-size 128 Phi-3-mini-4k-instruct-npu

:: Keep console open after execution
pause
exit