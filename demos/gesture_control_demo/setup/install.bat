@echo off
setlocal enabledelayedexpansion

:: Get the current directory where the script is placed
set "INSTALL_DIR=%CD%\openvino_build_deploy"

:: Check if Git is installed
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed. Please install Git and try again.
    exit /b
)

:: Clone the repository (if not already cloned)
if not exist "%INSTALL_DIR%" (
    echo Cloning repository...
    git clone https://github.com/openvinotoolkit/openvino_build_deploy.git "%INSTALL_DIR%"
) else (
    echo Repository already exists. Skipping cloning...
)

:: Navigate to Gesture Control Demo directory
cd /d "%INSTALL_DIR%\demos\gesture_control_demo"

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Final success message
echo.
echo ========================================
echo PalmPilot - Gesture Control Demo Ready!
echo ========================================
echo.
echo To start the demo:
echo   python main.py --stream 0
echo   python main.py --stream 1
echo.
echo ========================================
pause
exit
