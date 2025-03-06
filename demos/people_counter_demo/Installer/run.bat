@echo off
setlocal enabledelayedexpansion

:: Get the current directory where the script is placed
set "INSTALL_DIR=%CD%\openvino_build_deploy\demos\people_counter_demo"

:: Navigate to the People Counter Demo directory
cd /d "%INSTALL_DIR%"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install_people_counter.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Run the application
echo Running People Counter Demo...
python main.py --stream 0

:: Keep console open after execution
pause
exit
