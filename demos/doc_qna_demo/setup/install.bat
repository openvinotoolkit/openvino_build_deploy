@echo off
setlocal enabledelayedexpansion

:: Doc-QnA Demo - Windows 一键安装并运行

set "INSTALL_DIR=%CD%\openvino_build_deploy"

where git >nul 2>&1 || (echo ERROR: Git is not installed. & exit /b)
where python >nul 2>&1 || (echo ERROR: Python 3.10-3.12 is required. & exit /b)

if not exist "%INSTALL_DIR%" (
    echo Cloning repository...
    git clone https://github.com/openvinotoolkit/openvino_build_deploy.git "%INSTALL_DIR%"
) else (
    echo Repository exists. Pulling latest...
    cd /d "%INSTALL_DIR%" && git pull
)

cd /d "%INSTALL_DIR%\demos\doc_qna_demo"

:: Venv
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

:: Install + Run
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo.
echo Running Doc-QnA Demo...
python main.py

pause
exit
