"""
Data Publisher Service for sending traffic monitoring data to InfluxDB and MQTT
"""

from influxdb_client import InfluxDBClient, Point, WritePrecision
from datetime import datetime
import threading
import queue
import time
import json
import os
from PySide6.QtCore import QObject, Signal, QThread, QTimer

# Import MQTT Publisher
try:
    from .mqtt_publisher import MQTTPublisher
    MQTT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MQTT Publisher not available: {e}")
    MQTTPublisher = None
    MQTT_AVAILABLE = False

class DataPublisher(QObject):
    """
    Service to publish traffic monitoring data to InfluxDB and MQTT
    """
    
    def __init__(self, config_file=None):
        super().__init__()
        
        # Store config file path
        self.config_file = config_file
        
        # InfluxDB configuration
        self.url = "http://localhost:8086"
        self.token = "kNFfXEpPQoWrk5Tteowda21Dzv6xD3jY7QHSHHQHb5oYW6VH6mkAgX9ZMjQJkaHHa8FwzmyVFqDG7qqzxN09uQ=="
        self.org = "smart-intersection-org"
        self.bucket = "traffic_monitoring"
        
        # Load config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                influx_config = config.get('influxdb', {})
                self.url = influx_config.get('url', self.url)
                self.token = influx_config.get('token', self.token)
                self.org = influx_config.get('org', self.org)
                self.bucket = influx_config.get('bucket', self.bucket)
        
        # Initialize InfluxDB client
        self.client = None
        self.write_api = None
        self.connected = False
        
        # Initialize MQTT Publisher
        self.mqtt_publisher = None
        self.mqtt_enabled = False
        if MQTT_AVAILABLE:
            try:
                self.mqtt_publisher = MQTTPublisher(config_file)
                self.mqtt_enabled = True
                print("✅ MQTT Publisher integrated")
            except Exception as e:
                print(f"⚠️ Failed to initialize MQTT Publisher: {e}")
        
        # Data queue for batch processing
        self.data_queue = queue.Queue()
        
        # Device information
        self.device_id = "smart_intersection_001"
        self.camera_id = "camera_001"
        
        # Initialize connection
        self.connect_to_influxdb()
        
        # Start background thread for data publishing
        self.publisher_thread = threading.Thread(target=self._publisher_worker, daemon=True)
        self.publisher_thread.start()
        
        print("✅ DataPublisher initialized and ready (InfluxDB + MQTT)")
    
    def connect_to_influxdb(self):
        """Connect to InfluxDB"""
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            health = self.client.health()
            if health.status == "pass":
                self.write_api = self.client.write_api()
                self.connected = True
                print(f"✅ Connected to InfluxDB at {self.url}")
                
                # Send initial device info
                self.publish_device_info()
                return True
            else:
                print(f"❌ InfluxDB health check failed: {health.status}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to InfluxDB: {e}")
            return False
    
    def publish_device_info(self):
        """Publish device information to both InfluxDB and MQTT"""
        if not self.connected:
            return
            
        try:
            # InfluxDB point
            point = Point("device_info") \
                .tag("device_id", self.device_id) \
                .tag("camera_id", self.camera_id) \
                .field("status", "online") \
                .field("location", "Main Intersection") \
                .field("version", "1.0.0") \
                .time(datetime.utcnow())
            
            self.data_queue.put(point)
            
            # MQTT publishing
            if self.mqtt_enabled and self.mqtt_publisher:
                device_info = {
                    "location": "Main Intersection",
                    "version": "1.0.0"
                }
                self.mqtt_publisher.publish_device_status("online", device_info)
            
            print("📊 Device info queued for publishing")
        except Exception as e:
            print(f"❌ Error publishing device info: {e}")
    
    def publish_performance_data(self, fps, processing_time_ms, cpu_usage=None, gpu_usage=None):
        """Publish performance metrics to both InfluxDB and MQTT"""
        if not self.connected:
            return
            
        try:
            # InfluxDB point
            point = Point("performance") \
                .tag("device_id", self.device_id) \
                .tag("camera_id", self.camera_id) \
                .field("fps", float(fps)) \
                .field("processing_time_ms", float(processing_time_ms))
            
            if cpu_usage is not None:
                point = point.field("cpu_usage", float(cpu_usage))
            if gpu_usage is not None:
                point = point.field("gpu_usage", float(gpu_usage))
                
            point = point.time(datetime.utcnow())
            
            self.data_queue.put(point)
            
            # MQTT publishing
            if self.mqtt_enabled and self.mqtt_publisher:
                self.mqtt_publisher.publish_performance_data(fps, processing_time_ms, cpu_usage, gpu_usage)
            
            print(f"📊 Performance data queued: FPS={fps:.1f}, Processing={processing_time_ms:.1f}ms")
        except Exception as e:
            print(f"❌ Error publishing performance data: {e}")
    
    def publish_detection_events(self, vehicle_count, pedestrian_count=0):
        """Publish detection events to both InfluxDB and MQTT"""
        if not self.connected:
            return
            
        try:
            # InfluxDB point
            point = Point("detection_events") \
                .tag("device_id", self.device_id) \
                .tag("camera_id", self.camera_id) \
                .field("vehicle_count", int(vehicle_count)) \
                .field("pedestrian_count", int(pedestrian_count)) \
                .time(datetime.utcnow())
            
            self.data_queue.put(point)
            
            # MQTT publishing
            if self.mqtt_enabled and self.mqtt_publisher:
                self.mqtt_publisher.publish_detection_events(vehicle_count, pedestrian_count)
            
            print(f"📊 Detection events queued: Vehicles={vehicle_count}, Pedestrians={pedestrian_count}")
        except Exception as e:
            print(f"❌ Error publishing detection events: {e}")
    
    def publish_traffic_light_status(self, color, confidence=1.0):
        """Publish traffic light status to both InfluxDB and MQTT"""
        if not self.connected:
            return
            
        try:
            # Map colors to numeric values for Grafana
            color_numeric = {"red": 1, "yellow": 2, "green": 3, "unknown": 0}.get(color.lower(), 0)
            
            # InfluxDB point
            point = Point("traffic_light_status") \
                .tag("device_id", self.device_id) \
                .tag("camera_id", self.camera_id) \
                .tag("color", color.lower()) \
                .field("color_numeric", color_numeric) \
                .field("confidence", float(confidence)) \
                .time(datetime.utcnow())
            
            self.data_queue.put(point)
            
            # MQTT publishing
            if self.mqtt_enabled and self.mqtt_publisher:
                self.mqtt_publisher.publish_traffic_light_status(color, confidence)
            
            print(f"🚦 Traffic light status queued: {color.upper()} (confidence: {confidence:.2f})")
        except Exception as e:
            print(f"❌ Error publishing traffic light status: {e}")
    
    def publish_violation_event(self, violation_type, vehicle_id, details=None):
        """Publish violation events to both InfluxDB and MQTT"""
        if not self.connected:
            return
            
        try:
            # InfluxDB point
            point = Point("violation_events") \
                .tag("device_id", self.device_id) \
                .tag("camera_id", self.camera_id) \
                .tag("violation_type", violation_type) \
                .field("vehicle_id", str(vehicle_id)) \
                .field("count", 1)
            
            if details:
                if isinstance(details, dict):
                    for key, value in details.items():
                        if isinstance(value, (int, float, str)):
                            point = point.field(f"detail_{key}", value)
                        
            point = point.time(datetime.utcnow())
            
            self.data_queue.put(point)
            
            # MQTT publishing
            if self.mqtt_enabled and self.mqtt_publisher:
                self.mqtt_publisher.publish_violation_event(violation_type, vehicle_id, details)
            
            print(f"🚨 Violation event queued: {violation_type} - Vehicle {vehicle_id}")
        except Exception as e:
            print(f"❌ Error publishing violation event: {e}")
    
    def _publisher_worker(self):
        """Background worker to publish data in batches"""
        batch = []
        last_publish = time.time()
        
        while True:
            try:
                # Try to get data from queue (with timeout)
                try:
                    point = self.data_queue.get(timeout=1.0)
                    batch.append(point)
                except queue.Empty:
                    pass
                
                # Publish batch if we have data and enough time has passed or batch is full
                current_time = time.time()
                if batch and (len(batch) >= 10 or (current_time - last_publish) >= 5.0):
                    if self.connected and self.write_api:
                        try:
                            self.write_api.write(bucket=self.bucket, org=self.org, record=batch)
                            print(f"📤 Published batch of {len(batch)} data points to InfluxDB")
                            batch.clear()
                            last_publish = current_time
                        except Exception as e:
                            print(f"❌ Error writing batch to InfluxDB: {e}")
                            # Try to reconnect
                            self.connect_to_influxdb()
                
            except Exception as e:
                print(f"❌ Error in publisher worker: {e}")
                time.sleep(1)
    
    def get_status(self):
        """Get status of both InfluxDB and MQTT connections"""
        status = {
            'influxdb': {
                'connected': self.connected,
                'url': self.url,
                'org': self.org,
                'bucket': self.bucket
            },
            'mqtt': {
                'enabled': self.mqtt_enabled,
                'connected': False
            }
        }
        
        if self.mqtt_enabled and self.mqtt_publisher:
            mqtt_status = self.mqtt_publisher.get_connection_status()
            status['mqtt'].update(mqtt_status)
        
        return status
    
    def disconnect_all(self):
        """Disconnect from all services"""
        # Disconnect MQTT
        if self.mqtt_enabled and self.mqtt_publisher:
            self.mqtt_publisher.disconnect()
        
        # Close InfluxDB client
        if self.client:
            self.client.close()
            self.connected = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        # Publish device offline status
        if self.mqtt_enabled and self.mqtt_publisher:
            try:
                self.mqtt_publisher.publish_device_status("offline")
                self.mqtt_publisher.disconnect()
            except:
                pass
        
        # Close InfluxDB client
        if self.client:
            self.client.close()
