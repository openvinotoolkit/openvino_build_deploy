@echo off
setlocal enabledelayedexpansion

:: Get the current directory where the script is placed
set "INSTALL_DIR=%CD%\openvino_build_deploy\demos\gesture_control_demo"

:: Navigate to the Gesture Control Demo directory
cd /d "%INSTALL_DIR%"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Run the PalmPilot Gesture Control Demo
echo Running PalmPilot - Gesture Control Demo...
python main.py --stream 0

:: Keep console open after execution
pause
exit
