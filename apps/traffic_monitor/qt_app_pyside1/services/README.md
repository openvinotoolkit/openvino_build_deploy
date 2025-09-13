# Smart Intersection Services Configuration

## Service Configuration Files

This directory contains configuration files and setup scripts for the MQTT + InfluxDB + Grafana services integration.

### Directory Structure
```
services/
├── mqtt/
│   ├── mosquitto.conf
│   └── topics.json
├── influxdb/
│   ├── config.yml
│   └── init.flux
├── grafana/
│   ├── datasources/
│   ├── dashboards/
│   └── provisioning/
├── docker/
│   └── docker-compose.yml
└── scripts/
    ├── start_services.bat
    ├── stop_services.bat
    └── setup_services.py
```

### Service Ports
- **MQTT Broker**: 1883 (unsecured), 8883 (secured)
- **InfluxDB**: 8086 (HTTP API)
- **Grafana**: 3000 (Web UI)

### Quick Start
1. Run `scripts/setup_services.py` to download and configure services
2. Run `scripts/start_services.bat` to start all services
3. Access Grafana at http://localhost:3000 (admin/admin)
4. Desktop app will automatically connect to services
