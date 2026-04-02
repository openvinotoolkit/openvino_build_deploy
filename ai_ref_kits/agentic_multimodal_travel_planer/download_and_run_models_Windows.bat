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
set LLM_DEVICE_FROM_FLAG=
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
REM --llm-device: two argv, one argv with space, or --llm-device=VALUE (PowerShell often drops the value token)
set "ARG1=%~1"
if /I "!ARG1:~0,13!"=="--llm-device=" (
    set "LLM_DEVICE=!ARG1:~13!"
    set "LLM_DEVICE_FROM_FLAG=1"
    shift
    goto parse_args
)
REM Cannot GOTO from inside (...) blocks — use a flag and branch here
set "LLM_ARG_SHIFT1="
echo(!ARG1!| findstr /R /C:" " >nul && (
    for /f "tokens=1* delims= " %%a in ("!ARG1!") do (
        if /I "%%a"=="--llm-device" if not "%%b"=="" (
            set "LLM_DEVICE=%%b"
            set "LLM_DEVICE_FROM_FLAG=1"
            set "LLM_ARG_SHIFT1=1"
        )
    )
)
if defined LLM_ARG_SHIFT1 (
    set "LLM_ARG_SHIFT1="
    shift
    goto parse_args
)
if /I "%~1"=="--llm-device" (
    if "%~2"=="" (
        echo ERROR: --llm-device needs a value. Examples: --llm-device CPU   or   --llm-device=CPU
        exit /b 1
    )
    set "LLM_DEVICE=%~2"
    set "LLM_DEVICE_FROM_FLAG=1"
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

call :sanitize_device_var TARGET_DEVICE
call :sanitize_device_var LLM_DEVICE
call :sanitize_device_var VLM_DEVICE

REM Normalize device values once (cpu/gpu.0 -> CPU/GPU.0)
call :to_upper TARGET_DEVICE
call :to_upper LLM_DEVICE
call :to_upper VLM_DEVICE

REM Validate device values (must be CPU, GPU, or GPU.N) and ports (must be numeric)
call :validate_device TARGET_DEVICE "!TARGET_DEVICE!"
call :validate_device LLM_DEVICE "!LLM_DEVICE!"
call :validate_device VLM_DEVICE "!VLM_DEVICE!"
call :validate_port LLM_PORT "!LLM_PORT!"
call :validate_port VLM_PORT "!VLM_PORT!"

goto :after_validation_subroutines

:validate_device
REM %1 = variable name, %2 = current value (already normalized to upper-case)
set "DEV_NAME=%~1"
set "DEV_VALUE=%~2"

REM Allow empty value (use default behavior if any)
if "!DEV_VALUE!"=="" goto :validate_device_end

REM Whitespace-only (space/tab/NBSP etc.): clear the caller var and skip — avoids bad User env LLM_DEVICE
set "VDW="
set "VD_CH=!DEV_VALUE!"
for /f "delims=" %%Z in ('powershell -NoProfile -Command "if ([string]::IsNullOrWhiteSpace($env:VD_CH)) { Write-Output 1 } else { Write-Output 0 }"') do set "VDW=%%Z"
if "!VDW!"=="1" (
  set "%~1="
  goto :validate_device_end
)

REM Allow CPU or GPU directly
if /I "!DEV_VALUE!"=="CPU" goto :validate_device_end
if /I "!DEV_VALUE!"=="GPU" goto :validate_device_end

REM Check for GPU.N pattern
for /f "tokens=1,2 delims=." %%A in ("!DEV_VALUE!") do (
    set "DEV_PREFIX=%%A"
    set "DEV_SUFFIX=%%B"
)

if /I not "!DEV_PREFIX!"=="GPU" goto :invalid_device_value
if "!DEV_SUFFIX!"=="" goto :invalid_device_value

REM Ensure suffix is numeric
echo(!DEV_SUFFIX!| findstr /R "^[0-9][0-9]*$" >nul || goto :invalid_device_value
goto :validate_device_end

:invalid_device_value
echo.
echo ERROR: Invalid value for !DEV_NAME!: "!DEV_VALUE!"
echo        Supported formats: CPU, GPU, or GPU.N  (for example: GPU.0)
exit /b 1

:validate_device_end
set "DEV_NAME="
set "DEV_VALUE="
set "DEV_PREFIX="
set "DEV_SUFFIX="
goto :eof

:validate_port
REM %1 = variable name, %2 = current value
set "PORT_NAME=%~1"
set "PORT_VALUE=%~2"

REM Port must be non-empty and numeric
if "!PORT_VALUE!"=="" (
    echo.
    echo ERROR: Port variable !PORT_NAME! is empty.
    exit /b 1
)

echo(!PORT_VALUE!| findstr /R "^[0-9][0-9]*$" >nul || (
    echo.
    echo ERROR: Invalid value for !PORT_NAME!: "!PORT_VALUE!"
    echo        Port values must be numeric.
    exit /b 1
)

set "PORT_NAME="
set "PORT_VALUE="
goto :eof

:after_validation_subroutines
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

REM LLM device: --llm-device overrides auto TARGET_DEVICE for OVMS LLM only; otherwise inherit TARGET_DEVICE
if not defined LLM_DEVICE_FROM_FLAG (
    if "!LLM_DEVICE!"=="" set "LLM_DEVICE=!TARGET_DEVICE!"
)
if defined LLM_DEVICE_FROM_FLAG (
    if "!LLM_DEVICE!"=="" (
        echo ERROR: --llm-device needs a value ^(CPU, GPU, or GPU.N^).
        exit /b 1
    )
)
if "!VLM_DEVICE!"=="" set "VLM_DEVICE=!TARGET_DEVICE!"

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
REM Start-Process must use an argument array; a single -ArgumentList string is passed as ONE argv (OVMS would not see --target_device).
powershell -NoProfile -Command "$al=@('--port',$env:LLM_GRPC_PORT,'--rest_port',$env:LLM_PORT,'--model_repository_path',$env:MODELS_DIR,'--source_model',$env:LLM_MODEL,'--tool_parser','hermes3','--cache_size','0','--task','text_generation'); if (-not [string]::IsNullOrWhiteSpace($env:LLM_DEVICE)) { $al+='--target_device'; $al+=$env:LLM_DEVICE }; Start-Process -FilePath $env:OVMS_PATH -ArgumentList $al -RedirectStandardOutput ([IO.Path]::Combine($env:LOGS_DIR,'ovms_llm.log')) -RedirectStandardError ([IO.Path]::Combine($env:LOGS_DIR,'ovms_llm.err')) -WindowStyle Hidden" || (echo Failed to start LLM service && exit /b 1)
set "LLM_PID="
for /L %%n in (1,1,25) do (
  if "!LLM_PID!"=="" (
    ping -n 3 127.0.0.1 >nul
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%LLM_PORT%" ^| findstr "LISTENING"') do set "LLM_PID=%%a"
  )
)
if defined LLM_PID (echo LLM service started - PID: !LLM_PID!) else (echo LLM service started - still binding; check log: %LOGS_DIR%\ovms_llm.log)

REM Start VLM service
REM --port = gRPC, --rest_port = HTTP REST. REST on VLM_PORT for clients.
echo Starting VLM service (REST on %VLM_PORT%, gRPC on 8012)...
set VLM_GRPC_PORT=8012
REM Do not pass --target_device for VLM ^(MediaPipe^); OVMS expects device in model subconfig.json.
powershell -NoProfile -Command "$al=@('--port',$env:VLM_GRPC_PORT,'--rest_port',$env:VLM_PORT,'--model_name',$env:VLM_MODEL,'--model_path',$env:VLM_MODEL_PATH); Start-Process -FilePath $env:OVMS_PATH -ArgumentList $al -RedirectStandardOutput ([IO.Path]::Combine($env:LOGS_DIR,'ovms_vlm.log')) -RedirectStandardError ([IO.Path]::Combine($env:LOGS_DIR,'ovms_vlm.err')) -WindowStyle Hidden" || (echo Failed to start VLM service && exit /b 1)
set "VLM_PID="
REM VLM (vision) often binds slower than LLM on CPU CI; allow ~3 min of polling
for /L %%n in (1,1,90) do (
  if "!VLM_PID!"=="" (
    ping -n 3 127.0.0.1 >nul
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%VLM_PORT%" ^| findstr "LISTENING"') do set "VLM_PID=%%a"
  )
)
if defined VLM_PID (echo VLM service started - PID: !VLM_PID!) else (echo VLM service started - still binding; check log: %LOGS_DIR%\ovms_vlm.log)

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

:sanitize_device_var
setlocal EnableDelayedExpansion
set "_vn=%~1"
set "v=!%_vn%!"
if "!v!"=="" endlocal & exit /b 0
set "PSV=!v!"
for /f "delims=" %%A in ('powershell -NoProfile -Command "$s=$env:PSV; if ([string]::IsNullOrWhiteSpace($s)) { 'CLR' } else { $s.Trim() }"') do set "RES=%%A"
if "!RES!"=="CLR" (
  endlocal
  set "%~1="
  exit /b 0
)
for %%E in ("!RES!") do (
  endlocal
  set "%~1=%%~E"
)
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
call :sync_cfg_base

set "OVMS_MODEL_ID="
set "OVMS_ID_FILE=%TEMP%\ovms_model_id_%RANDOM%.txt"
if exist "%OVMS_ID_FILE%" del "%OVMS_ID_FILE%" >nul 2>&1

powershell -NoProfile -Command "$out=$env:OVMS_ID_FILE; $port=$env:LLM_PORT; $id=''; for ($i=0; $i -lt 30 -and -not $id; $i++) { try { $r=Invoke-RestMethod -Uri ('http://127.0.0.1:' + $port + '/v3/models') -Method Get -TimeoutSec 15; if ($r.data -and @($r.data).Count -gt 0 -and $r.data[0].id) { $id=[string]$r.data[0].id } } catch { } if (-not $id) { Start-Sleep -Seconds 3 } }; if ($id) { Set-Content -Path $out -Value $id -NoNewline }"
if exist "%OVMS_ID_FILE%" (
    set /p OVMS_MODEL_ID=<"%OVMS_ID_FILE%"
    del "%OVMS_ID_FILE%" >nul 2>&1
)

if not defined OVMS_MODEL_ID goto sync_cfg_no_model_id
call :sync_cfg_model_id
echo Synced agents model to OVMS model id: !OVMS_MODEL_ID!
goto sync_cfg_done

:sync_cfg_no_model_id
echo Note: LLM /v3/models not ready yet after wait - agents_config already has model and api_base from sync above.
echo       If agents fail to load the model, wait for OVMS then re-run this script or edit config\agents_config.yaml.

:sync_cfg_done

exit /b 0

:sync_cfg_base
powershell -NoProfile -Command "$cfg=$env:CONFIG_FILE; $port=$env:LLM_PORT; $llm=$env:LLM_MODEL; if (-not (Test-Path $cfg)) { exit 0 }; $q=[char]34; $text=Get-Content -Raw -Path $cfg; $text=[regex]::Replace($text, 'api_base:\s*\x22http://127\.0\.0\.1:[0-9]+/v3\x22', ('api_base: ' + $q + 'http://127.0.0.1:' + $port + '/v3' + $q)); $text=[regex]::Replace($text, 'model:\s*\x22openai:OpenVINO/[^\x22]*\x22', ('model: ' + $q + 'openai:' + $llm + $q)); Set-Content -Path $cfg -Value $text -Encoding UTF8"
exit /b 0

:sync_cfg_model_id
powershell -NoProfile -Command "$cfg=$env:CONFIG_FILE; $id=$env:OVMS_MODEL_ID; if (-not (Test-Path $cfg)) { exit 0 }; $q=[char]34; $text=Get-Content -Raw -Path $cfg; $text=[regex]::Replace($text, 'model:\s*\x22openai:[^\x22]*\x22', ('model: ' + $q + 'openai:' + $id + $q)); Set-Content -Path $cfg -Value $text -Encoding UTF8"
exit /b 0

:stop_only
echo Stopping existing processes on ports %LLM_PORT%, %VLM_PORT%, 8011, 8012...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":%LLM_PORT% :%VLM_PORT% :8011 :8012"') do (
    if not defined seen_%%a (
        set "seen_%%a=1"
        taskkill /F /PID %%a >nul 2>&1
    )
)
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
echo   --llm-device DEVICE    Device override for LLM ^(also --llm-device=CPU from PowerShell^)
echo   --vlm-device DEVICE    Not passed to VLM OVMS ^(use model subconfig.json^)
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 --device CPU
echo   %~nx0 --llm-port 9001 --vlm-port 9002
echo   %~nx0 --llm-device GPU.0 --vlm-device CPU
echo   %~nx0 --stop
endlocal
exit /b 0