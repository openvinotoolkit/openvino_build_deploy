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
echo Running chatbot demo
python chat_sample.py Phi-3-mini-4k-instruct-npu

:: Keep console open after execution
pause
exit