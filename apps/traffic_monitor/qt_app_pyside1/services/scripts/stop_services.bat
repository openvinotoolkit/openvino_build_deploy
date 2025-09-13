@echo off
REM Smart Intersection Services Stop Script
REM Stops all running services

echo ===============================================
echo Smart Intersection Services Shutdown
echo ===============================================

REM Check if Docker Compose services are running
docker-compose --version >nul 2>&1
if %errorlevel% equ 0 (
    cd /d "%~dp0..\docker"
    docker-compose ps >nul 2>&1
    if %errorlevel% equ 0 (
        echo 🛑 Stopping Docker Compose services...
        docker-compose down
        echo ✅ Docker services stopped
        goto :end
    )
)

echo 🛑 Stopping standalone services...

REM Stop Grafana (by finding and killing the process)
echo Stopping Grafana...
tasklist /FI "IMAGENAME eq grafana-server.exe" 2>NUL | find /I /N "grafana-server.exe">NUL
if "%ERRORLEVEL%"=="0" (
    taskkill /F /IM grafana-server.exe >nul 2>&1
    echo ✅ Grafana stopped
) else (
    echo ⚠️ Grafana was not running
)

REM Stop InfluxDB
echo Stopping InfluxDB...
tasklist /FI "IMAGENAME eq influxd.exe" 2>NUL | find /I /N "influxd.exe">NUL
if "%ERRORLEVEL%"=="0" (
    taskkill /F /IM influxd.exe >nul 2>&1
    echo ✅ InfluxDB stopped  
) else (
    echo ⚠️ InfluxDB was not running
)

REM Stop Mosquitto MQTT
echo Stopping Mosquitto MQTT...
tasklist /FI "IMAGENAME eq mosquitto.exe" 2>NUL | find /I /N "mosquitto.exe">NUL
if "%ERRORLEVEL%"=="0" (
    taskkill /F /IM mosquitto.exe >nul 2>&1
    echo ✅ Mosquitto MQTT stopped
) else (
    echo ⚠️ Mosquitto MQTT was not running
)

:end
echo.
echo ===============================================
echo All Services Stopped
echo ===============================================
echo.
echo 🛑 Smart Intersection services have been shut down
echo You can now safely close this window
echo.
pause
