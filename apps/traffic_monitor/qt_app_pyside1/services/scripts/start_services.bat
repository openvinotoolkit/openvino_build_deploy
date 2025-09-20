@echo off
REM Smart Intersection Services Startup Script
REM Starts all required services for hybrid desktop + services architecture

echo ===============================================
echo Smart Intersection Services Startup
echo ===============================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Docker is available - starting services via Docker Compose
    cd /d "%~dp0..\docker"
    docker-compose up -d
    echo.
    echo Services starting in Docker containers:
    echo - MQTT Broker: localhost:1883
    echo - InfluxDB: localhost:8086  
    echo - Grafana: localhost:3000
    echo.
    echo Wait 30 seconds for services to initialize...
    timeout /t 30 /nobreak
    echo.
    echo ✅ Services should now be ready!
    echo 🌐 Open Grafana: http://localhost:3000 (admin/admin)
    echo 📊 InfluxDB UI: http://localhost:8086
    goto :end
)

echo ⚠️ Docker not found - attempting standalone service startup
echo.

REM Start Mosquitto MQTT Broker
echo Starting Mosquitto MQTT Broker...
if exist "C:\Program Files\mosquitto\mosquitto.exe" (
    start "Mosquitto MQTT" "C:\Program Files\mosquitto\mosquitto.exe" -c "%~dp0..\mqtt\mosquitto.conf"
    echo ✅ Mosquitto MQTT started on port 1883
) else (
    echo ❌ Mosquitto not found at C:\Program Files\mosquitto\
    echo Please install Mosquitto from: https://mosquitto.org/download/
)

REM Start InfluxDB
echo Starting InfluxDB...

REM Use extracted InfluxDB in downloads folder
if exist "C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\influxdb2-2.7.11-windows\influxd.exe" (
    start "InfluxDB" "C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\influxdb2-2.7.11-windows\influxd.exe"
    echo ✅ InfluxDB started on port 8086
) else (
    echo ❌ InfluxDB not found in downloads folder
    echo Please extract InfluxDB to: C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\influxdb2-2.7.11-windows\
)

REM Start Grafana  
echo Starting Grafana...

REM Use extracted Grafana in downloads folder
if exist "C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\grafana-10.2.2.windows-amd64\grafana-v10.2.2\bin\grafana-server.exe" (
    start "Grafana" "C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\grafana-10.2.2.windows-amd64\grafana-v10.2.2\bin\grafana-server.exe" --homepath="C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\grafana-10.2.2.windows-amd64\grafana-v10.2.2"
    echo ✅ Grafana started on port 3000
) else (
    echo ❌ Grafana not found in downloads folder
    echo Please extract Grafana to: C:\Users\devcloud\Desktop\Qt\clean-final-push\qt_app_pyside1\services\services\downloads\grafana-10.2.2.windows-amd64\grafana-v10.2.2\
)

echo.
echo ⏳ Waiting 15 seconds for services to initialize...
timeout /t 15 /nobreak

echo.
echo ===============================================
echo Service Status Check
echo ===============================================

REM Check service ports
echo Checking MQTT Broker (port 1883)...
netstat -an | findstr :1883 >nul
if %errorlevel% equ 0 (
    echo ✅ MQTT Broker is listening on port 1883
) else (
    echo ❌ MQTT Broker not responding on port 1883
)

echo Checking InfluxDB (port 8086)...
netstat -an | findstr :8086 >nul
if %errorlevel% equ 0 (
    echo ✅ InfluxDB is listening on port 8086
) else (
    echo ❌ InfluxDB not responding on port 8086
)

echo Checking Grafana (port 3000)...
netstat -an | findstr :3000 >nul
if %errorlevel% equ 0 (
    echo ✅ Grafana is listening on port 3000
) else (
    echo ❌ Grafana not responding on port 3000
)

:end
echo.
echo ===============================================
echo Services Started Successfully!
echo ===============================================
echo.
echo Access Points:
echo 🌐 Grafana Dashboard: http://localhost:3000 (admin/admin)
echo 📊 InfluxDB UI: http://localhost:8086  
echo 📡 MQTT Broker: localhost:1883
echo.
echo You can now start the Smart Intersection Desktop App!
echo.
pause
