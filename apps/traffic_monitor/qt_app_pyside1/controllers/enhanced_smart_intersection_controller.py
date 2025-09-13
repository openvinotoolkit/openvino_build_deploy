"""
Enhanced Smart Intersection Controller with MQTT + InfluxDB Integration
Hybrid Desktop + Services architecture for professional monitoring
"""

import json
import time
import logging
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from PySide6.QtCore import QObject, Signal, QTimer, QThread

# Import the original controller
from controllers.smart_intersection_controller import SmartIntersectionController as BaseController


@dataclass 
class ServiceConfig:
    """Configuration for MQTT and InfluxDB services"""
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_keepalive: int = 60
    mqtt_client_id: str = "smart_intersection_desktop"
    
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "smart-intersection-super-secret-token"
    influxdb_org: str = "smart-intersection-org"
    influxdb_bucket: str = "traffic_monitoring"
    
    # Service health monitoring
    service_check_interval: int = 30  # seconds
    auto_reconnect: bool = True
    max_retries: int = 3


class MQTTPublisher(QObject):
    """Handles MQTT publishing for real-time events"""
    
    # Signals for connection status
    connected = Signal()
    disconnected = Signal()
    error_occurred = Signal(str)
    message_published = Signal(str, dict)
    
    def __init__(self, config: ServiceConfig):
        super().__init__()
        self.config = config
        self.client = None
        self.is_connected = False
        self.topic_base = "smartintersection"
        
        # Load topic definitions
        self.topics = self._load_topic_definitions()
        
        # Setup MQTT client
        self._setup_mqtt_client()
    
    def _load_topic_definitions(self) -> Dict:
        """Load MQTT topic definitions from config"""
        try:
            topics_path = Path(__file__).parent.parent / "services" / "mqtt" / "topics.json"
            if topics_path.exists():
                with open(topics_path, 'r') as f:
                    config = json.load(f)
                    return config.get('mqtt_topics', {}).get('topics', {})
        except Exception as e:
            print(f"Warning: Could not load MQTT topics config: {e}")
        
        # Default topics
        return {
            "detection": {"topic": "smartintersection/detection", "qos": 1},
            "violations": {"topic": "smartintersection/violations", "qos": 2},
            "performance": {"topic": "smartintersection/performance", "qos": 0},
            "traffic_flow": {"topic": "smartintersection/traffic/flow", "qos": 1},
            "pedestrian_safety": {"topic": "smartintersection/safety/pedestrian", "qos": 2},
            "roi_events": {"topic": "smartintersection/roi/events", "qos": 1},
            "system_health": {"topic": "smartintersection/system/health", "qos": 1}
        }
    
    def _setup_mqtt_client(self):
        """Setup MQTT client with callbacks"""
        self.client = mqtt.Client(client_id=self.config.mqtt_client_id)
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_message = self._on_message
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful MQTT connection"""
        if rc == 0:
            self.is_connected = True
            self.connected.emit()
            print("✅ MQTT: Connected to broker")
        else:
            self.error_occurred.emit(f"MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.is_connected = False
        self.disconnected.emit()
        print("🔌 MQTT: Disconnected from broker")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for successful message publish"""
        pass  # Could emit signal with message ID
    
    def _on_message(self, client, userdata, msg):
        """Callback for received messages"""
        pass  # Not used for publishing-only client
    
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        try:
            self.client.connect(
                self.config.mqtt_broker, 
                self.config.mqtt_port, 
                self.config.mqtt_keepalive
            )
            self.client.loop_start()
            return True
        except Exception as e:
            self.error_occurred.emit(f"MQTT connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish_detection_event(self, detection_data: Dict):
        """Publish object detection event"""
        self._publish_message("detection", detection_data)
    
    def publish_violation_event(self, violation_data: Dict):
        """Publish traffic violation event"""
        self._publish_message("violations", violation_data)
    
    def publish_performance_metrics(self, performance_data: Dict):
        """Publish system performance metrics"""
        self._publish_message("performance", performance_data)
    
    def publish_traffic_flow(self, flow_data: Dict):
        """Publish traffic flow analytics"""
        self._publish_message("traffic_flow", flow_data)
    
    def publish_pedestrian_safety(self, safety_data: Dict):
        """Publish pedestrian safety events"""
        self._publish_message("pedestrian_safety", safety_data)
    
    def publish_roi_event(self, roi_data: Dict):
        """Publish ROI-based events"""
        self._publish_message("roi_events", roi_data)
    
    def publish_system_health(self, health_data: Dict):
        """Publish system health status"""
        self._publish_message("system_health", health_data)
    
    def _publish_message(self, topic_key: str, data: Dict):
        """Internal method to publish messages"""
        if not self.is_connected:
            return False
        
        try:
            topic_config = self.topics.get(topic_key, {})
            topic = topic_config.get("topic", f"{self.topic_base}/{topic_key}")
            qos = topic_config.get("qos", 0)
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().isoformat()
            
            # Convert to JSON
            message = json.dumps(data)
            
            # Publish message
            result = self.client.publish(topic, message, qos=qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.message_published.emit(topic, data)
                return True
            else:
                self.error_occurred.emit(f"Failed to publish to {topic}: {result.rc}")
                return False
                
        except Exception as e:
            self.error_occurred.emit(f"Error publishing message: {e}")
            return False


class InfluxDBWriter(QObject):
    """Handles InfluxDB time series data writing"""
    
    # Signals for connection status
    connected = Signal()
    disconnected = Signal()
    error_occurred = Signal(str)
    data_written = Signal(str, int)  # measurement, points_count
    
    def __init__(self, config: ServiceConfig):
        super().__init__()
        self.config = config
        self.client = None
        self.write_api = None
        self.is_connected = False
        
        # Connect to InfluxDB
        self._connect()
    
    def _connect(self):
        """Connect to InfluxDB"""
        try:
            self.client = InfluxDBClient(
                url=self.config.influxdb_url,
                token=self.config.influxdb_token,
                org=self.config.influxdb_org
            )
            
            # Test connection
            self.client.ping()
            
            # Setup write API
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            
            self.is_connected = True
            self.connected.emit()
            print("✅ InfluxDB: Connected successfully")
            
        except Exception as e:
            self.is_connected = False
            self.error_occurred.emit(f"InfluxDB connection error: {e}")
            print(f"❌ InfluxDB: Connection failed - {e}")
    
    def disconnect(self):
        """Disconnect from InfluxDB"""
        if self.client:
            self.client.close()
            self.is_connected = False
            self.disconnected.emit()
    
    def write_performance_metrics(self, metrics: Dict):
        """Write performance metrics to InfluxDB"""
        if not self.is_connected:
            return
        
        try:
            point = Point("performance") \
                .tag("service", "desktop_app") \
                .field("fps", float(metrics.get("fps", 0))) \
                .field("gpu_usage", float(metrics.get("gpu_usage", 0))) \
                .field("memory_usage", float(metrics.get("memory_usage", 0))) \
                .field("processing_time_ms", float(metrics.get("processing_time_ms", 0))) \
                .field("camera_count", int(metrics.get("camera_count", 0))) \
                .field("active_objects", int(metrics.get("active_objects", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            self.write_api.write(bucket=self.config.influxdb_bucket, record=point)
            self.data_written.emit("performance", 1)
            
        except Exception as e:
            self.error_occurred.emit(f"Error writing performance metrics: {e}")
    
    def write_detection_event(self, detection: Dict):
        """Write detection event to InfluxDB"""
        if not self.is_connected:
            return
        
        try:
            camera_id = detection.get("camera_id", "unknown")
            camera_position = detection.get("camera_position", "unknown")
            
            point = Point("detection_events") \
                .tag("camera_id", camera_id) \
                .tag("camera_position", camera_position) \
                .tag("detection_model", detection.get("model", "yolo")) \
                .field("object_count", int(detection.get("object_count", 0))) \
                .field("vehicle_count", int(detection.get("vehicle_count", 0))) \
                .field("pedestrian_count", int(detection.get("pedestrian_count", 0))) \
                .field("confidence_avg", float(detection.get("confidence_avg", 0))) \
                .field("processing_time", float(detection.get("processing_time", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            self.write_api.write(bucket=self.config.influxdb_bucket, record=point)
            self.data_written.emit("detection_events", 1)
            
        except Exception as e:
            self.error_occurred.emit(f"Error writing detection event: {e}")
    
    def write_violation_event(self, violation: Dict):
        """Write violation event to InfluxDB"""
        if not self.is_connected:
            return
        
        try:
            point = Point("violation_events") \
                .tag("camera_id", violation.get("camera_id", "unknown")) \
                .tag("roi_id", violation.get("roi_id", "unknown")) \
                .tag("violation_category", violation.get("category", "unknown")) \
                .field("violation_type", violation.get("type", "")) \
                .field("severity_level", violation.get("severity", "")) \
                .field("object_id", violation.get("object_id", "")) \
                .field("confidence", float(violation.get("confidence", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            self.write_api.write(bucket=self.config.influxdb_bucket, record=point)
            self.data_written.emit("violation_events", 1)
            
        except Exception as e:
            self.error_occurred.emit(f"Error writing violation event: {e}")
    
    def write_traffic_flow(self, flow_data: Dict):
        """Write traffic flow data to InfluxDB"""
        if not self.is_connected:
            return
        
        try:
            point = Point("traffic_flow") \
                .tag("lane_id", flow_data.get("lane_id", "unknown")) \
                .tag("direction", flow_data.get("direction", "unknown")) \
                .tag("camera_position", flow_data.get("camera_position", "unknown")) \
                .field("vehicle_count", int(flow_data.get("vehicle_count", 0))) \
                .field("average_speed", float(flow_data.get("average_speed", 0))) \
                .field("lane_occupancy", float(flow_data.get("lane_occupancy", 0))) \
                .field("congestion_level", flow_data.get("congestion_level", "")) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            self.write_api.write(bucket=self.config.influxdb_bucket, record=point)
            self.data_written.emit("traffic_flow", 1)
            
        except Exception as e:
            self.error_occurred.emit(f"Error writing traffic flow: {e}")
    
    def write_roi_event(self, roi_event: Dict):
        """Write ROI event to InfluxDB"""
        if not self.is_connected:
            return
        
        try:
            point = Point("roi_events") \
                .tag("roi_id", roi_event.get("roi_id", "unknown")) \
                .tag("roi_type", roi_event.get("roi_type", "unknown")) \
                .tag("camera_id", roi_event.get("camera_id", "unknown")) \
                .field("event_type", roi_event.get("event_type", "")) \
                .field("dwell_time", float(roi_event.get("dwell_time", 0))) \
                .field("object_count", int(roi_event.get("object_count", 0))) \
                .field("activity_level", float(roi_event.get("activity_level", 0))) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            self.write_api.write(bucket=self.config.influxdb_bucket, record=point)
            self.data_written.emit("roi_events", 1)
            
        except Exception as e:
            self.error_occurred.emit(f"Error writing ROI event: {e}")


class ServiceHealthMonitor(QObject):
    """Monitors health of MQTT and InfluxDB services"""
    
    service_status_changed = Signal(str, bool)  # service_name, is_healthy
    all_services_healthy = Signal(bool)
    
    def __init__(self, config: ServiceConfig):
        super().__init__()
        self.config = config
        self.service_status = {
            "mqtt": False,
            "influxdb": False
        }
        
        # Setup monitoring timer
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self._check_services)
        self.monitor_timer.start(config.service_check_interval * 1000)
    
    def _check_services(self):
        """Check health of all services"""
        self._check_mqtt()
        self._check_influxdb()
        
        # Emit overall health status
        all_healthy = all(self.service_status.values())
        self.all_services_healthy.emit(all_healthy)
    
    def _check_mqtt(self):
        """Check MQTT broker health"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.config.mqtt_broker, self.config.mqtt_port))
            sock.close()
            
            is_healthy = result == 0
            if self.service_status["mqtt"] != is_healthy:
                self.service_status["mqtt"] = is_healthy
                self.service_status_changed.emit("mqtt", is_healthy)
                
        except Exception:
            if self.service_status["mqtt"]:
                self.service_status["mqtt"] = False
                self.service_status_changed.emit("mqtt", False)
    
    def _check_influxdb(self):
        """Check InfluxDB health"""
        try:
            import requests
            response = requests.get(f"{self.config.influxdb_url}/health", timeout=5)
            is_healthy = response.status_code == 200
            
            if self.service_status["influxdb"] != is_healthy:
                self.service_status["influxdb"] = is_healthy
                self.service_status_changed.emit("influxdb", is_healthy)
                
        except Exception:
            if self.service_status["influxdb"]:
                self.service_status["influxdb"] = False
                self.service_status_changed.emit("influxdb", False)


class EnhancedSmartIntersectionController(BaseController):
    """
    Enhanced Smart Intersection Controller with MQTT + InfluxDB integration
    Extends the base controller with hybrid desktop + services architecture
    """
    
    # Additional signals for service integration
    mqtt_connected = Signal()
    mqtt_disconnected = Signal()
    influxdb_connected = Signal()
    influxdb_disconnected = Signal()
    service_error = Signal(str, str)  # service_name, error_message
    services_healthy = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Load service configuration
        self.service_config = self._load_service_config()
        
        # Initialize service components
        self.mqtt_publisher = None
        self.influxdb_writer = None
        self.health_monitor = None
        
        # Service integration enabled flag
        self.services_enabled = True
        
        # Initialize services
        self._initialize_services()
        
        print("🚀 Enhanced Smart Intersection Controller initialized with services integration")
    
    def _load_service_config(self) -> ServiceConfig:
        """Load service configuration from file"""
        try:
            config_path = Path(__file__).parent.parent / "config" / "smart-intersection" / "services-config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                    return ServiceConfig(
                        mqtt_broker=config_data.get("mqtt", {}).get("broker", "localhost"),
                        mqtt_port=config_data.get("mqtt", {}).get("port", 1883),
                        influxdb_url=config_data.get("influxdb", {}).get("url", "http://localhost:8086"),
                        influxdb_token=config_data.get("influxdb", {}).get("token", "smart-intersection-super-secret-token"),
                        influxdb_org=config_data.get("influxdb", {}).get("org", "smart-intersection-org"),
                        influxdb_bucket=config_data.get("influxdb", {}).get("bucket", "traffic_monitoring")
                    )
        except Exception as e:
            print(f"Warning: Could not load service config: {e}")
        
        # Return default configuration
        return ServiceConfig()
    
    def _initialize_services(self):
        """Initialize MQTT and InfluxDB services"""
        if not self.services_enabled:
            return
        
        try:
            # Initialize MQTT Publisher
            self.mqtt_publisher = MQTTPublisher(self.service_config)
            self.mqtt_publisher.connected.connect(self.mqtt_connected.emit)
            self.mqtt_publisher.disconnected.connect(self.mqtt_disconnected.emit)
            self.mqtt_publisher.error_occurred.connect(lambda err: self.service_error.emit("mqtt", err))
            
            # Initialize InfluxDB Writer
            self.influxdb_writer = InfluxDBWriter(self.service_config)
            self.influxdb_writer.connected.connect(self.influxdb_connected.emit)
            self.influxdb_writer.disconnected.connect(self.influxdb_disconnected.emit)
            self.influxdb_writer.error_occurred.connect(lambda err: self.service_error.emit("influxdb", err))
            
            # Initialize Health Monitor
            self.health_monitor = ServiceHealthMonitor(self.service_config)
            self.health_monitor.all_services_healthy.connect(self.services_healthy.emit)
            
            # Connect to services
            if self.mqtt_publisher:
                self.mqtt_publisher.connect()
            
            print("✅ Services integration initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            self.services_enabled = False
    
    def set_services_enabled(self, enabled: bool):
        """Enable or disable services integration"""
        self.services_enabled = enabled
        
        if enabled and not self.mqtt_publisher:
            self._initialize_services()
        elif not enabled:
            self._disconnect_services()
    
    def _disconnect_services(self):
        """Disconnect from all services"""
        if self.mqtt_publisher:
            self.mqtt_publisher.disconnect()
        
        if self.influxdb_writer:
            self.influxdb_writer.disconnect()
    
    def process_frame(self, frame_data: Dict):
        """Override to add service integration"""
        # Call parent method
        result = super().process_frame(frame_data)
        
        # Publish to services if enabled
        if self.services_enabled and self.mqtt_publisher and self.mqtt_publisher.is_connected:
            self._publish_frame_data(frame_data, result)
        
        return result
    
    def _publish_frame_data(self, frame_data: Dict, processing_result: Dict):
        """Publish frame processing data to services"""
        try:
            # Extract detection data
            detections = frame_data.get('detections', [])
            camera_id = frame_data.get('camera_id', 'unknown')
            
            # Count objects by type
            vehicle_count = len([d for d in detections if d.get('class_name') in ['car', 'truck', 'bus']])
            pedestrian_count = len([d for d in detections if d.get('class_name') == 'person'])
            
            # Calculate average confidence
            confidences = [d.get('confidence', 0) for d in detections]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Detection event data
            detection_data = {
                "camera_id": camera_id,
                "camera_position": frame_data.get('camera_position', 'unknown'),
                "object_count": len(detections),
                "vehicle_count": vehicle_count,
                "pedestrian_count": pedestrian_count,
                "confidence_avg": avg_confidence,
                "processing_time": processing_result.get('processing_time_ms', 0),
                "frame_number": self.frame_count
            }
            
            # Publish detection event
            self.mqtt_publisher.publish_detection_event(detection_data)
            
            # Write to InfluxDB
            if self.influxdb_writer and self.influxdb_writer.is_connected:
                self.influxdb_writer.write_detection_event(detection_data)
            
        except Exception as e:
            print(f"Error publishing frame data: {e}")
    
    def _update_performance_stats(self):
        """Override to add service publishing"""
        super()._update_performance_stats()
        
        if not self.services_enabled:
            return
        
        try:
            # Get current performance data
            performance_data = self.analytics_data.get('performance', {})
            performance_data.update({
                "camera_count": len(self.scene_adapters),
                "active_objects": self.analytics_data.get('total_objects', 0),
                "timestamp": datetime.now().isoformat()
            })
            
            # Publish performance metrics
            if self.mqtt_publisher and self.mqtt_publisher.is_connected:
                self.mqtt_publisher.publish_performance_metrics(performance_data)
            
            # Write to InfluxDB
            if self.influxdb_writer and self.influxdb_writer.is_connected:
                self.influxdb_writer.write_performance_metrics(performance_data)
            
        except Exception as e:
            print(f"Error updating performance stats: {e}")
    
    def get_service_status(self) -> Dict:
        """Get status of all integrated services"""
        status = {
            "services_enabled": self.services_enabled,
            "mqtt": {
                "connected": self.mqtt_publisher.is_connected if self.mqtt_publisher else False,
                "broker": self.service_config.mqtt_broker,
                "port": self.service_config.mqtt_port
            },
            "influxdb": {
                "connected": self.influxdb_writer.is_connected if self.influxdb_writer else False,
                "url": self.service_config.influxdb_url,
                "bucket": self.service_config.influxdb_bucket
            },
            "health_monitor": {
                "active": self.health_monitor is not None
            }
        }
        
        return status
    
    def publish_violation_event(self, violation_data: Dict):
        """Publish traffic violation event"""
        if self.services_enabled and self.mqtt_publisher and self.mqtt_publisher.is_connected:
            self.mqtt_publisher.publish_violation_event(violation_data)
        
        if self.services_enabled and self.influxdb_writer and self.influxdb_writer.is_connected:
            self.influxdb_writer.write_violation_event(violation_data)
    
    def publish_traffic_flow_data(self, flow_data: Dict):
        """Publish traffic flow analytics"""
        if self.services_enabled and self.mqtt_publisher and self.mqtt_publisher.is_connected:
            self.mqtt_publisher.publish_traffic_flow(flow_data)
        
        if self.services_enabled and self.influxdb_writer and self.influxdb_writer.is_connected:
            self.influxdb_writer.write_traffic_flow(flow_data)
    
    def publish_pedestrian_safety_event(self, safety_data: Dict):
        """Publish pedestrian safety event"""
        if self.services_enabled and self.mqtt_publisher and self.mqtt_publisher.is_connected:
            self.mqtt_publisher.publish_pedestrian_safety(safety_data)
    
    def publish_roi_event(self, roi_data: Dict):
        """Publish ROI-based event"""
        if self.services_enabled and self.mqtt_publisher and self.mqtt_publisher.is_connected:
            self.mqtt_publisher.publish_roi_event(roi_data)
        
        if self.services_enabled and self.influxdb_writer and self.influxdb_writer.is_connected:
            self.influxdb_writer.write_roi_event(roi_data)
    
    def shutdown_services(self):
        """Gracefully shutdown all services"""
        print("🛑 Shutting down services integration...")
        
        if self.mqtt_publisher:
            self.mqtt_publisher.disconnect()
        
        if self.influxdb_writer:
            self.influxdb_writer.disconnect()
        
        if self.health_monitor:
            self.health_monitor.monitor_timer.stop()
        
        print("✅ Services integration shut down successfully")
