# Hybrid Desktop + Services Architecture Implementation Plan

## Overview
This implementation adds MQTT + InfluxDB + Grafana to create a professional-grade monitoring system alongside the existing PySide6 desktop application.

## Architecture
```
Desktop App (PySide6) → MQTT Broker → InfluxDB → Grafana Dashboard
       ↓                    ↓              ↓            ↓
Qt Signals         Real-time Events   Time Series    Rich Analytics
Local UI           MQTT Topics        Database       Visualizations
```

## Phase 1: Prerequisites & Downloads

### 1.1 Download Required Software (No Accounts Needed)

#### Mosquitto MQTT Broker
- **Download**: https://mosquitto.org/download/
- **File**: mosquitto-2.0.18-install-windows-x64.exe
- **Size**: ~8MB
- **Purpose**: Message broker for real-time event streaming

#### InfluxDB v2 
- **Download**: https://portal.influxdata.com/downloads/
- **File**: influxdb2-2.7.4-windows-amd64.zip
- **Size**: ~90MB  
- **Purpose**: Time series database for metrics storage

#### Grafana
- **Download**: https://grafana.com/grafana/download?platform=windows
- **File**: grafana-10.2.2.windows-amd64.zip
- **Size**: ~180MB
- **Purpose**: Analytics dashboard and visualization

### 1.2 Python Dependencies
```bash
pip install paho-mqtt influxdb-client grafana-api requests-async asyncio
```

## Phase 2: Service Installation

### 2.1 Install Mosquitto MQTT
1. Run mosquitto installer as Administrator
2. Install to default location: `C:\Program Files\mosquitto\`
3. Will create Windows service automatically
4. Default port: 1883 (unencrypted), 8883 (encrypted)

### 2.2 Setup InfluxDB
1. Extract InfluxDB zip to `C:\InfluxDB\`
2. Create data directory: `C:\InfluxDB\data\`
3. Will run as standalone service
4. Default port: 8086 (HTTP API)

### 2.3 Setup Grafana  
1. Extract Grafana zip to `C:\Grafana\`
2. Create configuration directory
3. Will run as standalone service
4. Default port: 3000 (Web UI)

## Phase 3: Configuration Files

### 3.1 Mosquitto Configuration
**File**: `C:\Program Files\mosquitto\mosquitto.conf`
```conf
# Smart Intersection MQTT Configuration
port 1883
listener 1883 0.0.0.0
allow_anonymous true
persistence true
persistence_location C:\Program Files\mosquitto\data\
log_dest file C:\Program Files\mosquitto\logs\mosquitto.log
log_type all
```

### 3.2 InfluxDB Configuration  
**File**: `C:\InfluxDB\config.yml`
```yaml
http-bind-address: ":8086"
storage-engine: tsm1
storage-directory: C:\InfluxDB\data
storage-wal-directory: C:\InfluxDB\wal
```

### 3.3 Grafana Configuration
**File**: `C:\Grafana\conf\defaults.ini` (modify)
```ini
[server]
http_port = 3000
domain = localhost

[security]
admin_user = admin
admin_password = admin

[database]
type = sqlite3
path = grafana.db
```

## Phase 4: Desktop Application Integration

### 4.1 MQTT Publisher Service
- Modify SmartIntersectionController to publish to MQTT topics
- Topics: `traffic/detection`, `traffic/violations`, `traffic/performance`
- Real-time event streaming

### 4.2 InfluxDB Writer Service  
- Background service to write metrics to InfluxDB
- Time series data: FPS, object counts, processing times
- Automatic batching and error handling

### 4.3 Grafana Dashboard Integration
- Pre-configured dashboards for traffic analytics
- Real-time charts and graphs
- Alerting system for violations

## Phase 5: Service Management

### 5.1 Windows Services Setup
- Create batch scripts to start/stop services
- Optional: Install as Windows services for auto-start
- Health monitoring and restart capabilities

### 5.2 Desktop Integration
- Add service status monitoring to desktop app
- Quick access buttons to open Grafana dashboards
- Service health indicators in UI

## Resource Usage Estimates
- **Mosquitto**: ~5-10MB RAM
- **InfluxDB**: ~50-100MB RAM  
- **Grafana**: ~100-150MB RAM
- **Total Additional**: ~200-300MB RAM

## Benefits
✅ Professional monitoring and analytics
✅ Real-time event streaming via MQTT
✅ Rich Grafana dashboards  
✅ Multiple device/remote access capability
✅ Time series data analysis
✅ Automated alerting system
✅ Scalable architecture

## Implementation Timeline
- **Phase 1-2**: 30 minutes (downloads + installation)
- **Phase 3**: 15 minutes (configuration)  
- **Phase 4**: 2 hours (desktop integration)
- **Phase 5**: 30 minutes (service management)
- **Total**: ~3-4 hours for complete implementation

## Next Steps
1. Download required software
2. Follow installation guide
3. Configure services
4. Integrate with desktop application
5. Create Grafana dashboards
6. Test end-to-end functionality
