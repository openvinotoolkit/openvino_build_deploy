@echo off
setlocal enabledelayedexpansion

echo === OpenVINO Model Server: setup + start (Baremetal) ===

REM Config
set OVMS_VERSION=v2025.4
set OVMS_DIR=%CD%\ovms
set MODELS_DIR=%CD%\models
set LOGS_DIR=%CD%\logs
set LLM_MODEL=OpenVINO/Qwen3-8B-int4-ov
set VLM_MODEL=OpenVINO/Phi-3.5-vision-instruct-int4-ov
set LLM_PORT=8001
set VLM_PORT=8002
set PYTHON_SUPPORT=python_on
set TARGET_DEVICE=

REM Download and extract OVMS package
if not exist "%OVMS_DIR%" (
    echo Downloading OpenVINO Model Server package...
    set PACKAGE_NAME=ovms_windows_%PYTHON_SUPPORT%.zip
    call set "DOWNLOAD_URL=https://github.com/openvinotoolkit/model_server/releases/download/%%OVMS_VERSION%%/%%PACKAGE_NAME%%"
    curl -L -o "%CD%\ovms.zip" "!DOWNLOAD_URL!" || (echo Failed to download package && exit /b 1)
    powershell -Command "Expand-Archive -Path '%CD%\ovms.zip' -DestinationPath '%CD%' -Force" 2>nul || (echo Failed to extract package && exit /b 1)
    del "%CD%\ovms.zip" >nul 2>&1
    echo Package extracted successfully.
)

REM Setup environment
set SETUP_SCRIPT=%OVMS_DIR%\setupvars.bat
if exist "%SETUP_SCRIPT%" (
    call "%SETUP_SCRIPT%" 2>nul || set PATH="%OVMS_DIR%;%PATH%"
) else (
    set PATH="%OVMS_DIR%;%PATH%"
)

REM Find OVMS binary
set OVMS_PATH=%OVMS_DIR%\ovms
if not exist "%OVMS_PATH%" set OVMS_PATH=%OVMS_DIR%\ovms.exe
if not exist "%OVMS_PATH%" (
    echo OVMS binary not found at: %OVMS_DIR%
    exit /b 1
)
echo OVMS binary found: %OVMS_PATH%

REM Detect GPU
set GPU_DETECTED=0
powershell -Command "$gpu = Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like '*Intel*' -and $_.AdapterRAM -gt 0} | Select-Object -First 1; if ($gpu) { exit 0 } else { exit 1 }" >nul 2>&1 && set GPU_DETECTED=1
if %GPU_DETECTED% equ 0 (
    wmic path win32_VideoController get name 2>nul | findstr /i /c:"intel" >nul 2>&1 && set GPU_DETECTED=1
)

REM Auto-select device
if "%TARGET_DEVICE%"=="" (
    if %GPU_DETECTED% equ 1 (set TARGET_DEVICE=GPU && echo Auto-selecting device: GPU) else (set TARGET_DEVICE=CPU && echo Auto-selecting device: CPU)
) else (
    echo Target device: %TARGET_DEVICE%
)

REM Install Python dependencies
if "%PYTHON_SUPPORT%"=="python_on" (
    pip3 install "Jinja2==3.1.6" "MarkupSafe==3.0.2" --quiet 2>nul || echo Warning: Failed to install Python dependencies
)

REM Create directories
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"

REM Download models
set LLM_MODEL_PATH=%MODELS_DIR%\%LLM_MODEL%
if not exist "%LLM_MODEL_PATH%" (
    echo Downloading LLM model: %LLM_MODEL%
    "%OVMS_PATH%" --pull --model_repository_path "%MODELS_DIR%" --source_model "%LLM_MODEL%" --task text_generation --tool_parser hermes3 || (echo Failed to download LLM model && exit /b 1)
) else (
    echo LLM model already exists, skipping download.
)

set VLM_MODEL_PATH=%MODELS_DIR%\%VLM_MODEL%
if not exist "%VLM_MODEL_PATH%" (
    echo Downloading VLM model: %VLM_MODEL%
    "%OVMS_PATH%" --pull --model_repository_path "%MODELS_DIR%" --source_model "%VLM_MODEL%" --task text_generation --pipeline_type VLM || (echo Failed to download VLM model && exit /b 1)
) else (
    echo VLM model already exists, skipping download.
)

REM Stop existing processes (LLM/VLM REST + gRPC ports)
echo Stopping existing processes on ports %LLM_PORT%, %VLM_PORT%, 8011, 8012...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LLM_PORT% :%VLM_PORT% :8011 :8012"') do taskkill /F /PID %%a >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start LLM service
REM --port = gRPC, --rest_port = HTTP REST (chat/completions). Agents use HTTP, so REST must be on LLM_PORT.
echo Starting LLM service (REST on %LLM_PORT%, gRPC on 8011)...
set LLM_GRPC_PORT=8011
set LLM_ARGS=--port %LLM_GRPC_PORT% --rest_port %LLM_PORT% --model_repository_path "%MODELS_DIR%" --source_model "%LLM_MODEL%" --tool_parser hermes3 --cache_size 2 --task text_generation --enable_prefix_caching true
if not "%TARGET_DEVICE%"=="" set LLM_ARGS=%LLM_ARGS% --target_device %TARGET_DEVICE%
start /B "" "%OVMS_PATH%" %LLM_ARGS% > "%LOGS_DIR%\ovms_llm.log" 2>&1 || (echo Failed to start LLM service && exit /b 1)
timeout /t 2 /nobreak >nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LLM_PORT%" ^| findstr "LISTENING"') do set LLM_PID=%%a
if defined LLM_PID (echo LLM service started - PID: %LLM_PID%) else (echo LLM service started - check log: %LOGS_DIR%\ovms_llm.log)

REM Start VLM service
REM --port = gRPC, --rest_port = HTTP REST. REST on VLM_PORT for clients.
echo Starting VLM service (REST on %VLM_PORT%, gRPC on 8012)...
set VLM_GRPC_PORT=8012
set VLM_ARGS=--port %VLM_GRPC_PORT% --rest_port %VLM_PORT% --model_name "%VLM_MODEL%" --model_path "%VLM_MODEL_PATH%"
start /B "" "%OVMS_PATH%" %VLM_ARGS% > "%LOGS_DIR%\ovms_vlm.log" 2>&1 || (echo Failed to start VLM service && exit /b 1)
timeout /t 2 /nobreak >nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%VLM_PORT%" ^| findstr "LISTENING"') do set VLM_PID=%%a
if defined VLM_PID (echo VLM service started - PID: %VLM_PID%) else (echo VLM service started - check log: %LOGS_DIR%\ovms_vlm.log)

timeout /t 3 /nobreak >nul

echo.
echo === OpenVINO Model Server is running ===
echo LLM endpoint : http://localhost:%LLM_PORT%
echo VLM endpoint : http://localhost:%VLM_PORT%
echo Log files: %LOGS_DIR%\ovms_llm.log, %LOGS_DIR%\ovms_vlm.log
if defined LLM_PID (echo To stop LLM: taskkill /F /PID %LLM_PID%)
if defined VLM_PID (echo To stop VLM: taskkill /F /PID %VLM_PID%)
echo.

endlocal