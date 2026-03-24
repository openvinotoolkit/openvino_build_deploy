@echo off
setlocal enabledelayedexpansion

echo === OpenVINO Model Server: setup + start (Baremetal) ===

REM Config
set OVMS_VERSION=v2026.0
set OVMS_DIR=%CD%\ovms
set MODELS_DIR=%CD%\models
set LOGS_DIR=%CD%\logs
set CONFIG_FILE=%~dp0config\agents_config.yaml
set LLM_MODEL=OpenVINO/Qwen3-8B-int4-ov
set VLM_MODEL=OpenVINO/Phi-3.5-vision-instruct-int4-ov
set LLM_PORT=8001
set VLM_PORT=8002
set PYTHON_SUPPORT=python_on
set TARGET_DEVICE=
set LLM_DEVICE=
set VLM_DEVICE=
set STOP_MODE=0

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-h" goto show_help
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-s" (
    set STOP_MODE=1
    shift
    goto parse_args
)
if /I "%~1"=="--stop" (
    set STOP_MODE=1
    shift
    goto parse_args
)
if /I "%~1"=="--llm-model" (
    set "LLM_MODEL=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--vlm-model" (
    set "VLM_MODEL=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--llm-port" (
    set "LLM_PORT=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--vlm-port" (
    set "VLM_PORT=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--models-dir" (
    set "MODELS_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--device" (
    set "TARGET_DEVICE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--llm-device" (
    set "LLM_DEVICE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--vlm-device" (
    set "VLM_DEVICE=%~2"
    shift
    shift
    goto parse_args
)

echo Unknown option: %~1
echo Run %~nx0 --help for usage.
exit /b 1

:args_done
if "%STOP_MODE%"=="1" goto stop_only

REM Normalize device values once (cpu/gpu.0 -> CPU/GPU.0)
call :to_upper TARGET_DEVICE
call :to_upper LLM_DEVICE
call :to_upper VLM_DEVICE

REM Download and extract OVMS package
if not exist "%OVMS_DIR%" (
    echo Downloading OpenVINO Model Server package...
    set PACKAGE_NAME=ovms_windows_!PYTHON_SUPPORT!.zip
    call set "DOWNLOAD_URL=https://github.com/openvinotoolkit/model_server/releases/download/!OVMS_VERSION!/!PACKAGE_NAME!"
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

REM Auto-select base device
if "%TARGET_DEVICE%"=="" (
    if %GPU_DETECTED% equ 1 (set TARGET_DEVICE=GPU && echo Auto-selecting device: GPU) else (set TARGET_DEVICE=CPU && echo Auto-selecting device: CPU)
) else (
    echo Base target device: %TARGET_DEVICE%
)

REM Resolve per-model devices
if "%LLM_DEVICE%"=="" set "LLM_DEVICE=%TARGET_DEVICE%"
if "%VLM_DEVICE%"=="" set "VLM_DEVICE=%TARGET_DEVICE%"

echo Configuration:
echo   LLM Model: %LLM_MODEL%
echo   VLM Model: %VLM_MODEL%
echo   LLM Port:  %LLM_PORT%
echo   VLM Port:  %VLM_PORT%
echo   LLM Device:%LLM_DEVICE%
echo   VLM Device:%VLM_DEVICE%
echo   Models Dir:%MODELS_DIR%
echo.

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
ping -n 3 127.0.0.1 >nul

REM Start LLM service
REM --port = gRPC, --rest_port = HTTP REST (chat/completions). Agents use HTTP, so REST must be on LLM_PORT.
echo Starting LLM service (REST on %LLM_PORT%, gRPC on 8011)...
set LLM_GRPC_PORT=8011
set LLM_ARGS=--port %LLM_GRPC_PORT% --rest_port %LLM_PORT% --model_repository_path "%MODELS_DIR%" --source_model "%LLM_MODEL%" --tool_parser hermes3 --cache_size 0 --task text_generation 
if not "%LLM_DEVICE%"=="" set LLM_ARGS=%LLM_ARGS% --target_device %LLM_DEVICE%
REM Use PowerShell Start-Process to launch detached
powershell -Command "Start-Process -FilePath '%OVMS_PATH%' -ArgumentList '%LLM_ARGS%' -RedirectStandardOutput '%LOGS_DIR%\ovms_llm.log' -RedirectStandardError '%LOGS_DIR%\ovms_llm.err' -WindowStyle Hidden" || (echo Failed to start LLM service && exit /b 1)
ping -n 3 127.0.0.1 >nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LLM_PORT%" ^| findstr "LISTENING"') do set LLM_PID=%%a
if defined LLM_PID (echo LLM service started - PID: %LLM_PID%) else (echo LLM service started - check log: %LOGS_DIR%\ovms_llm.log)

REM Start VLM service
REM --port = gRPC, --rest_port = HTTP REST. REST on VLM_PORT for clients.
echo Starting VLM service (REST on %VLM_PORT%, gRPC on 8012)...
set VLM_GRPC_PORT=8012
set VLM_ARGS=--port %VLM_GRPC_PORT% --rest_port %VLM_PORT% --model_name "%VLM_MODEL%" --model_path "%VLM_MODEL_PATH%"
if not "%VLM_DEVICE%"=="" set VLM_ARGS=%VLM_ARGS% --target_device %VLM_DEVICE%
REM Use PowerShell Start-Process to launch detached
powershell -Command "Start-Process -FilePath '%OVMS_PATH%' -ArgumentList '%VLM_ARGS%' -RedirectStandardOutput '%LOGS_DIR%\ovms_vlm.log' -RedirectStandardError '%LOGS_DIR%\ovms_vlm.err' -WindowStyle Hidden" || (echo Failed to start VLM service && exit /b 1)
ping -n 3 127.0.0.1 >nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%VLM_PORT%" ^| findstr "LISTENING"') do set VLM_PID=%%a
if defined VLM_PID (echo VLM service started - PID: %VLM_PID%) else (echo VLM service started - check log: %LOGS_DIR%\ovms_vlm.log)

call :sync_agents_config

ping -n 4 127.0.0.1 >nul

echo.
echo === OpenVINO Model Server is running ===
echo LLM endpoint : http://localhost:%LLM_PORT%
echo VLM endpoint : http://localhost:%VLM_PORT%
echo Log files: %LOGS_DIR%\ovms_llm.log, %LOGS_DIR%\ovms_vlm.log
if defined LLM_PID (echo To stop LLM: taskkill /F /PID %LLM_PID%)
if defined VLM_PID (echo To stop VLM: taskkill /F /PID %VLM_PID%)
echo.

endlocal
exit /b 0

:to_upper
setlocal enabledelayedexpansion
set "_var_name=%~1"
set "_value=!%_var_name%!"
if "!_value!"=="" (
    endlocal & exit /b 0
)
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$args[0].ToUpperInvariant()" "!_value!"`) do set "_upper=%%I"
endlocal & set "%~1=%_upper%"
exit /b 0

:sync_agents_config
if not exist "%CONFIG_FILE%" (
    echo Warning: config file not found: %CONFIG_FILE%
    exit /b 0
)

echo Syncing config\agents_config.yaml with current LLM settings...
powershell -NoProfile -Command "$cfg=$env:CONFIG_FILE; $port=$env:LLM_PORT; $llm=$env:LLM_MODEL; if (-not (Test-Path $cfg)) { exit 0 }; $text=Get-Content -Raw -Path $cfg; $text=[regex]::Replace($text, 'api_base:\s*\x22http://127\.0\.0\.1:[0-9]+/v3\x22', ('api_base: `"http://127.0.0.1:' + $port + '/v3`"')); $text=[regex]::Replace($text, 'model:\s*\x22openai:OpenVINO/[^\x22]*\x22', ('model: `"openai:' + $llm + '`"')); Set-Content -Path $cfg -Value $text -Encoding UTF8"

set "OVMS_MODEL_ID="
set "OVMS_ID_FILE=%TEMP%\ovms_model_id_%RANDOM%.txt"
if exist "%OVMS_ID_FILE%" del "%OVMS_ID_FILE%" >nul 2>&1

powershell -NoProfile -Command "$id=''; try { $r=Invoke-RestMethod -Uri ('http://127.0.0.1:' + $env:LLM_PORT + '/v3/models') -Method Get -TimeoutSec 5; if ($r.data -and $r.data.Count -gt 0) { $id = $r.data[0].id } } catch {}; if ($id) { Set-Content -Path $env:OVMS_ID_FILE -Value $id -NoNewline }"
if exist "%OVMS_ID_FILE%" (
    set /p OVMS_MODEL_ID=<"%OVMS_ID_FILE%"
    del "%OVMS_ID_FILE%" >nul 2>&1
)

if defined OVMS_MODEL_ID (
    powershell -NoProfile -Command "$cfg=$env:CONFIG_FILE; $id=$env:OVMS_MODEL_ID; if (-not (Test-Path $cfg)) { exit 0 }; $text=Get-Content -Raw -Path $cfg; $text=[regex]::Replace($text, 'model:\s*\x22openai:[^\x22]*\x22', ('model: `"openai:' + $id + '`"')); Set-Content -Path $cfg -Value $text -Encoding UTF8"
    echo Synced agents model to OVMS model id: !OVMS_MODEL_ID!
) else (
    echo Warning: could not resolve OVMS model id from /v3/models
)

exit /b 0

:stop_only
echo Stopping existing processes on ports %LLM_PORT%, %VLM_PORT%, 8011, 8012...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LLM_PORT% :%VLM_PORT% :8011 :8012"') do taskkill /F /PID %%a >nul 2>&1
echo Done.
endlocal
exit /b 0

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   -h, --help             Show this help message
echo   -s, --stop             Stop running OVMS processes
echo   --llm-model MODEL      LLM model ^(default: %LLM_MODEL%^)
echo   --vlm-model MODEL      VLM model ^(default: %VLM_MODEL%^)
echo   --llm-port PORT        LLM REST port ^(default: %LLM_PORT%^)
echo   --vlm-port PORT        VLM REST port ^(default: %VLM_PORT%^)
echo   --models-dir DIR       Models directory ^(default: %MODELS_DIR%^)
echo   --device DEVICE        Base device for both models ^(CPU, GPU, GPU.0, ...^)
echo   --llm-device DEVICE    Device override for LLM
echo   --vlm-device DEVICE    Device override for VLM
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 --device CPU
echo   %~nx0 --llm-port 9001 --vlm-port 9002
echo   %~nx0 --llm-device GPU.0 --vlm-device CPU
echo   %~nx0 --stop
endlocal
exit /b 0