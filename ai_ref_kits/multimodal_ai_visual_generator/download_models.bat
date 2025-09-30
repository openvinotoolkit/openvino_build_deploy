@echo off
setlocal enabledelayedexpansion

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat


huggingface-cli download OpenVINO/Qwen2.5-7B-Instruct-int4-ov --local-dir models/Qwen2.5-7B-Instruct-INT4
huggingface-cli download OpenVINO/FLUX.1-schnell-int4-ov --local-dir models/FLUX.1-schnell-INT4

:: Keep console open after execution
pause
exit
