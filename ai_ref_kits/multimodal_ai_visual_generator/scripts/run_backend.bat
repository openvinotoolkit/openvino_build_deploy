@echo off
setlocal enabledelayedexpansion

pushd ..

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Run the application
echo Running StoryTeller backend...

SET IMAGE_MODEL_TYPE=FLUX.1-schnell
SET LLM_MODEL_TYPE=Qwen2.5-7B-Instruct
SET MODEL_PRECISION=INT4
uvicorn main:app --host 0.0.0.0 --port 8000

:: Keep console open after execution
pause
exit
