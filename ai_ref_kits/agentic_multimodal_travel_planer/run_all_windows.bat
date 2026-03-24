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

set "MODEL_ARGS=%MODEL_ARGS% %1"
shift
goto parse_args

:collect_model_args
if "%~1"=="" goto args_done
set "MODEL_ARGS=%MODEL_ARGS% %1"
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

if "%STOP_MODE%"=="1" (
    echo Stopping unified stack...
    call :run_python "start_agents.py" --stop
    call :run_python "start_mcp_servers.py" --stop --kill
    call :stop_ovms_ports
    echo All stop commands sent.
    exit /b 0
)

if "%SKIP_MODELS%"=="0" (
    echo.
    echo === Step 1/4: Starting OVMS models ===
    call "%MODELS_SCRIPT%" %MODEL_ARGS%
    if errorlevel 1 (
        echo ERROR: Model startup failed.
        exit /b 1
    )
) else (
    echo Skipping OVMS model startup.
)

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
echo Model options:
echo   Any additional options are passed to download_and_run_models_Windows.bat
echo   (for example --llm-model, --vlm-model, --llm-port, --vlm-port, --device).
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 --device CPU --llm-port 9001 --vlm-port 9002
echo   %~nx0 --stop
echo   %~nx0 --skip-ui
exit /b 0
