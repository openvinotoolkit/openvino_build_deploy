@echo off
setlocal enabledelayedexpansion

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
pip install openvino==2025.1.0 ultralytics==8.3.120

:: Final success message
echo.
echo ========================================
echo All requirements installed!
echo ========================================
pause
exit