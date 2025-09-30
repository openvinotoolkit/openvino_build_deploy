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
pip install -r requirements.txt

:: Final success message
echo.
echo ========================================
echo All requirements installed for Storyteller.
echo You can now run the demo:
echo 1. download_models.bat
echo 2. run_backend.bat
echo 3. run_frontend.bat
echo ========================================
pause
exit
