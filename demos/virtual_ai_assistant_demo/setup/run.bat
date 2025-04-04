@echo off
setlocal enabledelayedexpansion

:: Get the current directory where the script is placed
set "INSTALL_DIR=%CD%\openvino_build_deploy\demos\virtual_ai_assistant_demo"

:: Navigate to the Virtual AI Assistant Demo directory
cd /d "%INSTALL_DIR%"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Run the application
echo Running Virtual AI Assistant Demo...
python main.py --public

:: Keep console open after execution
pause
exit
