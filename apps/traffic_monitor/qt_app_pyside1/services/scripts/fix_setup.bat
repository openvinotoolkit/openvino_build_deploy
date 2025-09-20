@echo off
echo ======================================================================
echo 🔧 SMART INTERSECTION SERVICES - QUICK FIX
echo ======================================================================

echo.
echo 🛠️ Fixing setup issues...

echo.
echo 1️⃣ Installing Mosquitto MQTT Broker...
echo    📌 Please install manually as administrator:
echo    📂 Location: C:\Users\devcloud\Desktop\Clean\clean-final-push\qt_app_pyside1\services\services\downloads\mosquitto-2.0.18-install-windows-x64.exe
echo    📍 Install to: C:\Program Files\mosquitto\
echo.
echo    Or run this command as administrator:
echo    C:\Users\devcloud\Desktop\Clean\clean-final-push\qt_app_pyside1\services\services\downloads\mosquitto-2.0.18-install-windows-x64.exe /S
echo.

echo 2️⃣ Manual InfluxDB Setup Required:
echo    🌐 Download from: https://github.com/influxdata/influxdb/releases/latest
echo    📁 Extract to: C:\SmartIntersectionServices\influxdb\
echo    💡 Look for file like: influxdb2-2.7.11-windows-amd64.zip
echo.

echo 3️⃣ Grafana is ready! ✅
echo    📂 Already extracted to: C:\SmartIntersectionServices\grafana\
echo.

echo 4️⃣ After manual installs, run:
echo    📜 start_services.bat
echo.

echo ======================================================================
echo 📋 MANUAL INSTALLATION STEPS:
echo ======================================================================
echo.
echo Step 1: Install Mosquitto
echo    • Right-click PowerShell/CMD and "Run as administrator"
echo    • Run: C:\Users\devcloud\Desktop\Clean\clean-final-push\qt_app_pyside1\services\services\downloads\mosquitto-2.0.18-install-windows-x64.exe
echo    • Use default installation path: C:\Program Files\mosquitto\
echo.
echo Step 2: Install InfluxDB
echo    • Download: https://github.com/influxdata/influxdb/releases/download/v2.7.11/influxdb2-2.7.11-windows-amd64.zip
echo    • Extract to: C:\SmartIntersectionServices\influxdb\
echo    • Should contain: influxd.exe and other files
echo.
echo Step 3: Test Services
echo    • Run: start_services.bat
echo    • Check: http://localhost:3000 (Grafana)
echo    • Check: http://localhost:8086 (InfluxDB)
echo    • MQTT: localhost:1883
echo.

pause
