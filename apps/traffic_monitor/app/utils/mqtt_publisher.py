"""
MQTT Publisher Service for Smart Intersection System
Real-time messaging for traffic monitoring data
"""

import json
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, Optional
import paho.mqtt.client as mqtt
from PySide6.QtCore import QObject, Signal
import logging

class MQTTPublisher(QObject):
    """
    MQTT Publisher for real-time traffic monitoring data
    Publishes to predefined topics for different data types
    """
    
    # Qt Signals for connection status
    connected = Signal()
    disconnected = Signal()
    message_published = Signal(str, str)  # topic, message
    connection_error = Signal(str)
    
    def __init__(self, config_file=None):
        super().__init__()
        
        # MQTT Configuration
        self.broker_host = "localhost"
        self.broker_port = 1883
        self.client_id = "smart_intersection_publisher"
        self.username = None
        self.password = None
        self.keepalive = 60
        
        # Load configuration if provided
        if config_file:
            self._load_config(config_file)
        
        # MQTT Client setup
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        self.client.on_log = self._on_log
        
        # Connection state
        self.is_connected = False
        self.connection_attempts = 0
        self.max_retries = 5
        
        # Message queue for reliable delivery
        self.message_queue = queue.Queue()
        self.publishing_enabled = True
        
        # Device identification
        self.device_id = "smart_intersection_001"
        self.camera_id = "camera_001"
        
        # Topic definitions (matching services/mqtt/topics.json)
        self.topics = {
            'detection_vehicles': 'detection/vehicles',
            'detection_pedestrians': 'detection/pedestrians',
            'violations_red_light': 'violations/red_light',
            'violations_speed': 'violations/speed',
            'violations_crosswalk': 'violations/crosswalk',
            'violations_wrong_way': 'violations/wrong_way',
            'performance_fps': 'performance/fps',
            'performance_system': 'performance/system',
            'traffic_light_status': 'traffic/light_status',
            'device_status': 'device/status',
            'system_alerts': 'system/alerts'
        }
        
        # Start background publisher thread
        self.publisher_thread = threading.Thread(target=self._publisher_worker, daemon=True)
        self.publisher_thread.start()
        
        # Initialize connection
        self.connect()
        
        print("✅ MQTT Publisher initialized")
    
    def _load_config(self, config_file):
        """Load MQTT configuration from file"""
        try:
            import os
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    mqtt_config = config.get('mqtt', {})
                    self.broker_host = mqtt_config.get('broker_host', self.broker_host)
                    self.broker_port = mqtt_config.get('broker_port', self.broker_port)
                    self.username = mqtt_config.get('username')
                    self.password = mqtt_config.get('password')
                    self.client_id = mqtt_config.get('client_id', self.client_id)
        except Exception as e:
            print(f"⚠️ Could not load MQTT config: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            
            print(f"🔄 Attempting to connect to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.client.connect_async(self.broker_host, self.broker_port, self.keepalive)
            self.client.loop_start()
            
        except Exception as e:
            print(f"❌ MQTT connection error: {e}")
            self.connection_error.emit(str(e))
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.publishing_enabled = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful MQTT connection"""
        if rc == 0:
            self.is_connected = True
            self.connection_attempts = 0
            print(f"✅ Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            self.connected.emit()
            
            # Publish device online status
            self.publish_device_status("online")
            
        else:
            self.is_connected = False
            print(f"❌ MQTT connection failed with code {rc}")
            self._handle_connection_error(rc)
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        self.is_connected = False
        print(f"⚠️ Disconnected from MQTT broker (code: {rc})")
        self.disconnected.emit()
        
        if rc != 0:  # Unexpected disconnection
            self._handle_reconnection()
    
    def _on_publish(self, client, userdata, mid):
        """Callback for successful message publish"""
        # Optional: Track published messages
        pass
    
    def _on_log(self, client, userdata, level, buf):
        """MQTT client logging"""
        if level == mqtt.MQTT_LOG_ERR:
            print(f"🔴 MQTT Error: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            print(f"🟡 MQTT Warning: {buf}")
    
    def _handle_connection_error(self, rc):
        """Handle connection errors with retry logic"""
        error_messages = {
            1: "Incorrect protocol version",
            2: "Invalid client identifier", 
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        
        error_msg = error_messages.get(rc, f"Unknown error {rc}")
        print(f"❌ MQTT Connection failed: {error_msg}")
        self.connection_error.emit(error_msg)
        
        self._handle_reconnection()
    
    def _handle_reconnection(self):
        """Handle automatic reconnection"""
        if self.connection_attempts < self.max_retries:
            self.connection_attempts += 1
            delay = min(2 ** self.connection_attempts, 30)  # Exponential backoff, max 30s
            print(f"🔄 Reconnecting in {delay}s (attempt {self.connection_attempts}/{self.max_retries})")
            
            def reconnect():
                time.sleep(delay)
                if self.publishing_enabled:
                    self.connect()
            
            threading.Thread(target=reconnect, daemon=True).start()
        else:
            print("❌ Max reconnection attempts reached")
    
    def _create_message(self, data: Dict[str, Any]) -> str:
        """Create standardized MQTT message"""
        message = {
            'timestamp': datetime.utcnow().isoformat(),
            'device_id': self.device_id,
            'camera_id': self.camera_id,
            **data
        }
        return json.dumps(message, default=str)
    
    def _publish_message(self, topic: str, message: str, qos: int = 1):
        """Internal method to publish message"""
        if not self.is_connected:
            # Queue message for later delivery
            self.message_queue.put((topic, message, qos))
            return False
        
        try:
            result = self.client.publish(topic, message, qos=qos)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📤 Published to {topic}: {len(message)} bytes")
                self.message_published.emit(topic, message)
                return True
            else:
                print(f"❌ Failed to publish to {topic}: {result.rc}")
                return False
        except Exception as e:
            print(f"❌ Error publishing to {topic}: {e}")
            return False
    
    def _publisher_worker(self):
        """Background worker to handle queued messages"""
        while self.publishing_enabled:
            try:
                # Process queued messages when connected
                if self.is_connected and not self.message_queue.empty():
                    try:
                        topic, message, qos = self.message_queue.get(timeout=0.1)
                        self._publish_message(topic, message, qos)
                    except queue.Empty:
                        pass
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"❌ Error in MQTT publisher worker: {e}")
                time.sleep(1)
    
    # Public Publishing Methods
    
    def publish_detection_events(self, vehicle_count: int, pedestrian_count: int = 0):
        """Publish vehicle and pedestrian detection events"""
        # Vehicle detections
        if vehicle_count > 0:
            vehicle_data = {
                'event_type': 'vehicle_detection',
                'count': vehicle_count,
                'location': 'main_intersection'
            }
            message = self._create_message(vehicle_data)
            self._publish_message(self.topics['detection_vehicles'], message)
        
        # Pedestrian detections
        if pedestrian_count > 0:
            pedestrian_data = {
                'event_type': 'pedestrian_detection', 
                'count': pedestrian_count,
                'location': 'crosswalk_area'
            }
            message = self._create_message(pedestrian_data)
            self._publish_message(self.topics['detection_pedestrians'], message)
    
    def publish_violation_event(self, violation_type: str, vehicle_id: str, details: Optional[Dict] = None):
        """Publish traffic violation events"""
        topic_map = {
            'red_light_violation': 'violations_red_light',
            'speed_violation': 'violations_speed', 
            'crosswalk_violation': 'violations_crosswalk',
            'wrong_way_violation': 'violations_wrong_way'
        }
        
        topic_key = topic_map.get(violation_type, 'system_alerts')
        
        violation_data = {
            'event_type': 'violation',
            'violation_type': violation_type,
            'vehicle_id': vehicle_id,
            'severity': 'high',
            'location': 'main_intersection'
        }
        
        if details:
            violation_data.update(details)
        
        message = self._create_message(violation_data)
        self._publish_message(self.topics[topic_key], message)
    
    def publish_performance_data(self, fps: float, processing_time_ms: float, 
                                cpu_usage: Optional[float] = None, gpu_usage: Optional[float] = None):
        """Publish system performance metrics"""
        # FPS data
        fps_data = {
            'metric_type': 'fps',
            'value': fps,
            'unit': 'frames_per_second'
        }
        fps_message = self._create_message(fps_data)
        self._publish_message(self.topics['performance_fps'], fps_message)
        
        # System performance data
        system_data = {
            'metric_type': 'system_performance',
            'processing_time_ms': processing_time_ms
        }
        
        if cpu_usage is not None:
            system_data['cpu_usage'] = cpu_usage
        if gpu_usage is not None:
            system_data['gpu_usage'] = gpu_usage
        
        system_message = self._create_message(system_data)
        self._publish_message(self.topics['performance_system'], system_message)
    
    def publish_traffic_light_status(self, color: str, confidence: float = 1.0):
        """Publish traffic light status changes"""
        light_data = {
            'event_type': 'traffic_light_change',
            'color': color.lower(),
            'confidence': confidence,
            'intersection_id': 'main_intersection'
        }
        
        message = self._create_message(light_data)
        self._publish_message(self.topics['traffic_light_status'], message)
    
    def publish_device_status(self, status: str, additional_info: Optional[Dict] = None):
        """Publish device status updates"""
        device_data = {
            'event_type': 'device_status',
            'status': status,
            'version': '1.0.0',
            'location': 'main_intersection'
        }
        
        if additional_info:
            device_data.update(additional_info)
        
        message = self._create_message(device_data)
        self._publish_message(self.topics['device_status'], message)
    
    def publish_system_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Publish system alerts and notifications"""
        alert_data = {
            'event_type': 'system_alert',
            'alert_type': alert_type,
            'message': message,
            'severity': severity
        }
        
        alert_message = self._create_message(alert_data)
        self._publish_message(self.topics['system_alerts'], alert_message)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'broker_host': self.broker_host,
            'broker_port': self.broker_port,
            'client_id': self.client_id,
            'connection_attempts': self.connection_attempts,
            'queued_messages': self.message_queue.qsize()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.disconnect()
