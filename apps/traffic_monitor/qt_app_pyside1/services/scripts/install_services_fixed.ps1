# Smart Intersection Services - Simple Installation Script
# Run this as Administrator for best results

Write-Host "======================================================================"
Write-Host "🚦 SMART INTERSECTION SERVICES - AUTOMATED INSTALLATION"
Write-Host "======================================================================"

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "⚠️  Warning: Not running as administrator" -ForegroundColor Yellow
    Write-Host "   Some installations may fail" -ForegroundColor Yellow
    Write-Host ""
}

# Paths
$servicesDir = "C:\Users\devcloud\Desktop\Clean\clean-final-push\qt_app_pyside1\services"
$downloadsDir = "$servicesDir\services\downloads"
$installDir = "C:\SmartIntersectionServices"

Write-Host "🛠️ Starting automated installation..." -ForegroundColor Green
Write-Host ""

# 1. Install Mosquitto MQTT Broker
Write-Host "1️⃣ Installing Mosquitto MQTT Broker..." -ForegroundColor White
$mosquittoInstaller = "$downloadsDir\mosquitto-2.0.18-install-windows-x64.exe"

if (Test-Path $mosquittoInstaller) {
    Write-Host "   📦 Found installer: $mosquittoInstaller" -ForegroundColor Green
    if ($isAdmin) {
        Write-Host "   🔧 Installing Mosquitto silently..." -ForegroundColor Yellow
        try {
            # Try multiple silent install arguments
            $silentArgs = @("/S", "/silent", "/verysilent")
            $installed = $false
            
            foreach ($arg in $silentArgs) {
                try {
                    Start-Process -FilePath $mosquittoInstaller -ArgumentList $arg -Wait -NoNewWindow -ErrorAction Stop
                    Write-Host "   ✅ Mosquitto installed successfully with $arg!" -ForegroundColor Green
                    $installed = $true
                    break
                } catch {
                    Write-Host "   ⚠️ Silent install with $arg failed, trying next..." -ForegroundColor Yellow
                }
            }
            
            if (-not $installed) {
                throw "All silent install methods failed"
            }
        } catch {
            Write-Host "   ❌ Installation failed" -ForegroundColor Red
            Write-Host "   📌 Please run installer manually: $mosquittoInstaller" -ForegroundColor Yellow
        }
    } else {
        Write-Host "   📌 Starting installer (requires manual approval)..." -ForegroundColor Yellow
        Start-Process -FilePath $mosquittoInstaller -Wait
    }
} else {
    Write-Host "   ❌ Installer not found: $mosquittoInstaller" -ForegroundColor Red
}

Write-Host ""

# 2. Download and install InfluxDB
Write-Host "2️⃣ Installing InfluxDB..." -ForegroundColor White
$influxDir = "$installDir\influxdb"
$influxExe = "$influxDir\influxd.exe"

if (-not (Test-Path $influxExe)) {
    Write-Host "   📥 Downloading InfluxDB v2.7.11..." -ForegroundColor Yellow
    $influxUrl = "https://github.com/influxdata/influxdb/releases/download/v2.7.11/influxdb2-2.7.11-windows-amd64.zip"
    $influxZip = "$downloadsDir\influxdb2-2.7.11-windows-amd64.zip"
    
    try {
        # Download InfluxDB using modern method
        Invoke-WebRequest -Uri $influxUrl -OutFile $influxZip -ErrorAction Stop
        Write-Host "   ✅ Downloaded InfluxDB" -ForegroundColor Green
        
        # Extract InfluxDB
        Write-Host "   📦 Extracting InfluxDB..." -ForegroundColor Yellow
        if (-not (Test-Path $influxDir)) { 
            New-Item -ItemType Directory -Path $influxDir -Force | Out-Null 
        }
        Expand-Archive -Path $influxZip -DestinationPath $influxDir -Force -ErrorAction Stop
        
        # Move files from subdirectory if needed
        $subDirs = Get-ChildItem -Path $influxDir -Directory
        if ($subDirs.Count -gt 0) {
            $subDir = $subDirs[0]
            Get-ChildItem -Path $subDir.FullName -Recurse | Move-Item -Destination $influxDir -Force
            Remove-Item -Path $subDir.FullName -Recurse -Force
        }
        
        Write-Host "   ✅ InfluxDB extracted to: $influxDir" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ InfluxDB installation failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   📌 Please download manually from: $influxUrl" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ✅ InfluxDB already installed: $influxExe" -ForegroundColor Green
}

Write-Host ""

# 3. Check and setup Grafana
Write-Host "3️⃣ Setting up Grafana..." -ForegroundColor White
$grafanaDir = "$installDir\grafana"
$grafanaExe = "$grafanaDir\bin\grafana-server.exe"
$grafanaZip = "$downloadsDir\grafana-10.2.2.windows-amd64.zip"

if (-not (Test-Path $grafanaExe)) {
    if (Test-Path $grafanaZip) {
        Write-Host "   📦 Extracting Grafana..." -ForegroundColor Yellow
        try {
            if (-not (Test-Path $grafanaDir)) { 
                New-Item -ItemType Directory -Path $grafanaDir -Force | Out-Null 
            }
            Expand-Archive -Path $grafanaZip -DestinationPath $grafanaDir -Force -ErrorAction Stop
            
            # Move files from subdirectory if needed
            $subDirs = Get-ChildItem -Path $grafanaDir -Directory
            if ($subDirs.Count -gt 0) {
                $subDir = $subDirs[0]
                Get-ChildItem -Path $subDir.FullName -Recurse | Move-Item -Destination $grafanaDir -Force
                Remove-Item -Path $subDir.FullName -Recurse -Force
            }
            
            Write-Host "   ✅ Grafana extracted to: $grafanaDir" -ForegroundColor Green
        } catch {
            Write-Host "   ❌ Grafana extraction failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "   ❌ Grafana zip not found: $grafanaZip" -ForegroundColor Red
        Write-Host "   📌 Download from: https://grafana.com/grafana/download?platform=windows" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ✅ Grafana already ready: $grafanaExe" -ForegroundColor Green
}

Write-Host ""

# 4. Test installations
Write-Host "4️⃣ Testing installations..." -ForegroundColor White

$mosquittoExe = "C:\Program Files\mosquitto\mosquitto.exe"
if (Test-Path $mosquittoExe) {
    Write-Host "   ✅ Mosquitto: $mosquittoExe" -ForegroundColor Green
} else {
    Write-Host "   ❌ Mosquitto not found: $mosquittoExe" -ForegroundColor Red
}

if (Test-Path $influxExe) {
    Write-Host "   ✅ InfluxDB: $influxExe" -ForegroundColor Green
} else {
    Write-Host "   ❌ InfluxDB not found: $influxExe" -ForegroundColor Red
}

if (Test-Path $grafanaExe) {
    Write-Host "   ✅ Grafana: $grafanaExe" -ForegroundColor Green
} else {
    Write-Host "   ❌ Grafana not found: $grafanaExe" -ForegroundColor Red
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "🎉 Installation Complete!" -ForegroundColor Green
Write-Host "======================================================================"
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor White
Write-Host "   1. Run: start_services.bat" -ForegroundColor Yellow
Write-Host "   2. Open: http://localhost:3000 (Grafana - admin/admin)" -ForegroundColor Yellow
Write-Host "   3. Open: http://localhost:8086 (InfluxDB)" -ForegroundColor Yellow
Write-Host "   4. MQTT available at: localhost:1883" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Services may take 30-60 seconds to fully start" -ForegroundColor Cyan

Read-Host "Press Enter to continue..."
