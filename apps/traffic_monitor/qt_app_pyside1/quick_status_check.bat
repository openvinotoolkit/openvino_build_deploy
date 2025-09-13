@echo off
REM Quick Service Status Check for Smart Intersection System
REM Checks if MQTT, InfluxDB, and Grafana are running

echo ===============================================
echo Smart Intersection - Service Status Check
echo ===============================================
echo Time: %date% %time%
echo.

echo 🔍 Checking Service Ports...
echo ============================================

REM Check MQTT Broker (port 1883)
echo 📡 MQTT Broker (port 1883):
netstat -an | findstr :1883 >nul
if %errorlevel% equ 0 (
    echo    ✅ RUNNING - MQTT Broker is listening
    echo    🌐 Connect: mqtt://localhost:1883
) else (
    echo    ❌ STOPPED - MQTT Broker not responding
    echo    💡 Run: start_services.bat
)
echo.

REM Check InfluxDB (port 8086)
echo 📊 InfluxDB (port 8086):
netstat -an | findstr :8086 >nul
if %errorlevel% equ 0 (
    echo    ✅ RUNNING - InfluxDB is listening
    echo    🌐 Web UI: http://localhost:8086
) else (
    echo    ❌ STOPPED - InfluxDB not responding
    echo    💡 Run: start_services.bat
)
echo.

REM Check Grafana (port 3000)
echo 📈 Grafana (port 3000):
netstat -an | findstr :3000 >nul
if %errorlevel% equ 0 (
    echo    ✅ RUNNING - Grafana is listening
    echo    🌐 Dashboard: http://localhost:3000 (admin/admin)
) else (
    echo    ❌ STOPPED - Grafana not responding
    echo    💡 Run: start_services.bat
)
echo.

echo 🔍 Checking Service Processes...
echo ==========================================

REM Check for MQTT process
tasklist | findstr /i mosquitto >nul
if %errorlevel% equ 0 (
    echo 📡 Mosquitto process: ✅ RUNNING
) else (
    echo 📡 Mosquitto process: ❌ NOT FOUND
)

REM Check for InfluxDB process
tasklist | findstr /i influx >nul
if %errorlevel% equ 0 (
    echo 📊 InfluxDB process: ✅ RUNNING
) else (
    echo 📊 InfluxDB process: ❌ NOT FOUND
)

REM Check for Grafana process
tasklist | findstr /i grafana >nul
if %errorlevel% equ 0 (
    echo 📈 Grafana process: ✅ RUNNING
) else (
    echo 📈 Grafana process: ❌ NOT FOUND
)

echo.
echo ===============================================
echo Quick Actions:
echo ===============================================
echo 🚀 Start all services:    start_services.bat
echo 🛑 Stop all services:     stop_services.bat
echo 🔍 Detailed status:       python check_system_status.py
echo 📊 Test InfluxDB:         python check_influxdb_status.py
echo 📡 Test MQTT:             python check_mqtt_status.py
echo 📈 Test Grafana:          python check_grafana_status.py
echo.

pause
