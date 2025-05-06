@echo off
setlocal enabledelayedexpansion

:: Get the current directory where the script is placed
set "INSTALL_DIR=%CD%\openvino_build_deploy\demos\paint_your_dreams_demo"

:: Navigate to the demo directory
cd /d "%INSTALL_DIR%"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo To choose a different model, run: run.bat --menu

:: Set default model
set "model=OpenVINO/LCM_Dreamshaper_v7-fp16-ov"

:: Check for --menu flag
if "%1"=="--menu" (
    echo.
    echo Choose a model to run:
    echo [1] OpenVINO/LCM_Dreamshaper_v7-int8-ov
    echo [2] OpenVINO/LCM_Dreamshaper_v7-fp16-ov
    echo [3] dreamlike-art/dreamlike-anime-1.0
    echo [4] OpenVINO/FLUX.1-schnell-int4-ov
    echo [5] OpenVINO/FLUX.1-schnell-int8-ov
    echo [6] OpenVINO/FLUX.1-schnell-fp16-ov
    set /p modelChoice="Enter model number (default is 2): "

    if "!modelChoice!"=="1" (
        set "model=OpenVINO/LCM_Dreamshaper_v7-int8-ov"
    ) else if "!modelChoice!"=="2" (
        set "model=OpenVINO/LCM_Dreamshaper_v7-fp16-ov"
    ) else if "!modelChoice!"=="3" (
        set "model=dreamlike-art/dreamlike-anime-1.0"
    ) else if "!modelChoice!"=="4" (
        set "model=OpenVINO/FLUX.1-schnell-int4-ov"
    ) else if "!modelChoice!"=="5" (
        set "model=OpenVINO/FLUX.1-schnell-int8-ov"
    ) else if "!modelChoice!"=="6" (
        set "model=OpenVINO/FLUX.1-schnell-fp16-ov"
    ) else (
        echo No valid selection. Using default model: !model!
    )

    echo.
    echo Running Paint Your Dreams Demo with model: !model! (public sharing enabled)
    python main.py --model_name "!model!" --public
    goto end
)

:: Default behavior (no --menu)
echo.
echo Running Paint Your Dreams Demo with default model: !model!
python main.py --model_name "!model!" --public

:end
:: Keep console open after execution
pause
exit