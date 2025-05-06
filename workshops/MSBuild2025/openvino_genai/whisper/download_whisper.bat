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
echo Download and Export Whisper Base Model
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base-npu --disable-stateful

:: Keep console open after execution
pause
exit