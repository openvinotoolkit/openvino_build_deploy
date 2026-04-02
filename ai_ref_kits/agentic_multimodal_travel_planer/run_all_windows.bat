@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "MODELS_SCRIPT=%SCRIPT_DIR%download_and_run_models_Windows.bat"
set "MCP_SCRIPT=%SCRIPT_DIR%start_mcp_servers.py"
set "AGENTS_SCRIPT=%SCRIPT_DIR%start_agents.py"
set "UI_SCRIPT=%SCRIPT_DIR%start_ui.py"

set "STOP_MODE=0"
set "SKIP_MODELS=0"
set "SKIP_MCP=0"
set "SKIP_AGENTS=0"
set "SKIP_UI=0"
set "MODEL_ARGS="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-h" goto show_help
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="--stop" (
    set "STOP_MODE=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-models" (
    set "SKIP_MODELS=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-mcp" (
    set "SKIP_MCP=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-agents" (
    set "SKIP_AGENTS=1"
    shift
    goto parse_args
)
if /I "%~1"=="--skip-ui" (
    set "SKIP_UI=1"
    shift
    goto parse_args
)

if /I "%~1"=="--" (
    shift
    goto collect_model_args
)

set "MODEL_ARGS=!MODEL_ARGS! %~1"
shift
goto parse_args

:collect_model_args
if "%~1"=="" goto args_done
set "MODEL_ARGS=!MODEL_ARGS! %~1"
shift
goto collect_model_args

:args_done
if not exist "%MODELS_SCRIPT%" (
    echo ERROR: Missing %MODELS_SCRIPT%
    exit /b 1
)
if not exist "%MCP_SCRIPT%" (
    echo ERROR: Missing %MCP_SCRIPT%
    exit /b 1
)
if not exist "%AGENTS_SCRIPT%" (
    echo ERROR: Missing %AGENTS_SCRIPT%
    exit /b 1
)
if not exist "%UI_SCRIPT%" (
    echo ERROR: Missing %UI_SCRIPT%
    exit /b 1
)

cd /d "%SCRIPT_DIR%"

REM Do not put "call ... %MODEL_ARGS%" inside parenthesized IF blocks: %% vars expand at block parse time
REM and MODEL_ARGS is often still empty then, so --llm-device etc. never reach the model script.
REM Fresh cmd.exe parses argv for the model script (fixes missing args when the parent was started from PowerShell).
if "%STOP_MODE%"=="1" goto do_stop_stack
goto after_stop_stack
:do_stop_stack
echo Stopping unified stack...
echo Stopping OVMS models...
cmd.exe /c call "%MODELS_SCRIPT%" --stop!MODEL_ARGS!
echo Stopping agents...
call :run_python "start_agents.py" --stop
echo Stopping MCP servers...
call :run_python "start_mcp_servers.py" --stop --kill
echo All stop commands sent.
exit /b 0
:after_stop_stack

if "%SKIP_MODELS%"=="1" goto skip_models_step
echo.
echo === Step 1/4: Starting OVMS models ===
cmd.exe /c call "%MODELS_SCRIPT%"!MODEL_ARGS!
if errorlevel 1 (
    echo ERROR: Model startup failed.
    echo OVMS logs: %SCRIPT_DIR%logs\ovms_llm.err  %SCRIPT_DIR%logs\ovms_vlm.err
    exit /b 1
)
goto after_models_step
:skip_models_step
echo Skipping OVMS model startup.
:after_models_step

if "%SKIP_MCP%"=="0" (
    echo.
    echo === Step 2/4: Starting MCP servers ===
    call :run_python "start_mcp_servers.py"
    if errorlevel 1 (
        echo ERROR: MCP startup failed.
        exit /b 1
    )
) else (
    echo Skipping MCP startup.
)

if "%SKIP_AGENTS%"=="0" (
    echo.
    echo === Step 3/4: Starting agents ===
    call :run_python "start_agents.py"
    if errorlevel 1 (
        echo ERROR: Agent startup failed.
        exit /b 1
    )
) else (
    echo Skipping agents startup.
)

if "%SKIP_UI%"=="0" (
    echo.
    echo === Step 4/4: Starting UI ===
    echo Launching Gradio UI in foreground ^(close window or Ctrl+C to stop UI^).
    call :run_python "start_ui.py"
    exit /b !errorlevel!
) else (
    echo Skipping UI startup.
)

exit /b 0

:run_python
where python >nul 2>&1
if not errorlevel 1 (
    python %*
    exit /b !errorlevel!
)
where py >nul 2>&1
if not errorlevel 1 (
    py -3 %*
    exit /b !errorlevel!
)
where python3 >nul 2>&1
if not errorlevel 1 (
    python3 %*
    exit /b !errorlevel!
)
echo ERROR: python, py, or python3 not found in PATH.
exit /b 1

:stop_ovms_ports
echo Stopping OVMS processes listening on ports 8001, 8002, 8011, 8012...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8001 :8002 :8011 :8012"') do taskkill /F /PID %%a >nul 2>&1
exit /b 0

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Unified launcher for:
echo   1) download_and_run_models_Windows.bat
echo   2) start_mcp_servers.py
echo   3) start_agents.py
echo   4) start_ui.py
echo.
echo Options:
echo   -h, --help      Show this help message
echo   --stop          Stop agents, MCP servers, and OVMS processes
echo   --skip-models   Skip OVMS model startup
echo   --skip-mcp      Skip MCP startup
echo   --skip-agents   Skip agent startup
echo   --skip-ui       Skip UI startup
echo.
echo Model options ^(forwarded to download_and_run_models_Windows.bat^):
echo   --llm-model MODEL     LLM model id ^(e.g. OpenVINO/Qwen3-8B-int4-ov^)
echo   --vlm-model MODEL     VLM model id
echo   --llm-port PORT       LLM REST port
echo   --vlm-port PORT       VLM REST port
echo   --models-dir DIR      Models directory
echo   --device DEVICE       Base device for LLM/VLM defaults ^(CPU, GPU, GPU.0, ...^)
echo   --llm-device DEVICE   LLM-only device ^(overrides --device for LLM^)
echo                           From PowerShell use --llm-device=CPU if the value token is dropped.
echo   --vlm-device DEVICE   Recorded for docs; VLM OVMS uses model subconfig.json
echo   Optional --            End launcher parsing; remaining tokens go to the model script
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 --device CPU --llm-port 9001 --vlm-port 9002
echo   %~nx0 --llm-model OpenVINO/Qwen3-8B-int4-ov --llm-device GPU.0
echo   %~nx0 --vlm-model OpenVINO/Phi-3.5-vision-instruct-int4-ov --vlm-device CPU
echo   %~nx0 --stop
echo   %~nx0 --skip-ui
exit /b 0
