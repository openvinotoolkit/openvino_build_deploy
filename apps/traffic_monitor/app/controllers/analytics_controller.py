from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
from collections import defaultdict, deque
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

class AnalyticsController(QObject):
    """
    Controller for traffic analytics and statistics.
    
    Manages:
    - Vehicle counts by class
    - Violation statistics
    - Temporal analytics (traffic over time)
    - Speed statistics
    """
    analytics_updated = Signal(dict)  # Emitted when analytics are updated
    
    def __init__(self):
        """Initialize the analytics controller"""
        super().__init__()
        
        # Detection statistics
        self.detection_counts = defaultdict(int)
        self.detection_history = []
        
        # Violation statistics
        self.violation_counts = defaultdict(int)
        self.violation_history = []
        
        # Time series data (for charts)
        self.time_series = {
            'timestamps': [],
            'vehicle_counts': [],
            'pedestrian_counts': [],
            'violation_counts': []
        }
        
        # Performance metrics
        self.fps_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Aggregated metrics
        self.aggregated_metrics = {
            'total_vehicles': 0,
            'total_pedestrians': 0,
            'total_violations': 0,
            'avg_processing_time': 0,
            'avg_fps': 0,
            'peak_vehicle_count': 0,
            'peak_violation_hour': None
        }
        
        # Initialize current time window
        self.current_window = datetime.now().replace(
            minute=0, second=0, microsecond=0
        )
        self.window_stats = defaultdict(int)
        
        # Add traffic light analytics
        self.traffic_light_counts = defaultdict(int)  # Counts by color
        self.traffic_light_color_series = []  # List of (timestamp, color)
        self.traffic_light_color_numeric = []  # For charting: 0=unknown, 1=red, 2=yellow, 3=green
        self.traffic_light_color_map = {'unknown': 0, 'red': 1, 'yellow': 2, 'green': 3}
        
        self._last_update = time.time()
    @Slot(object, list, float)
    def process_frame_data(self, frame, detections, metrics):
        """
        Process frame data for analytics.
        
        Args:
            frame: Video frame
            detections: List of detections
            metrics: Dictionary containing metrics like 'detection_fps' or directly the fps value
        """
        try:
            # Empty violations list since violation detection is disabled
            violations = []
            
            # Debug info
            det_count = len(detections) if detections else 0
            print(f"Analytics processing: {det_count} detections")
        except Exception as e:
            print(f"Error in process_frame_data initialization: {e}")
            violations = []
        # Update FPS history - safely handle different metrics formats
        try:
            if isinstance(metrics, dict):
                fps = metrics.get('detection_fps', None)
                if isinstance(fps, (int, float)):
                    self.fps_history.append(fps)
            elif isinstance(metrics, (int, float)):
                # Handle case where metrics is directly the fps value
                self.fps_history.append(metrics)
            else:
                # Fallback if metrics is neither dict nor numeric
                print(f"Warning: Unexpected metrics type: {type(metrics)}")
        except Exception as e:
            print(f"Error processing metrics: {e}")
            # Add a default value to keep analytics running
            self.fps_history.append(0.0)
        
        # Process detections
        vehicle_count = 0
        pedestrian_count = 0
        
        # --- Traffic light analytics ---
        traffic_light_count = 0
        traffic_light_colors = []
        for det in detections:
            class_name = det.get('class_name', 'unknown').lower()
            self.detection_counts[class_name] += 1
            
            # Track vehicles vs pedestrians
            if class_name in ['car', 'truck', 'bus', 'motorcycle']:
                vehicle_count += 1
            elif class_name == 'person':
                pedestrian_count += 1
            if class_name in ['traffic light', 'trafficlight', 'tl', 'signal']:
                traffic_light_count += 1
                color = det.get('traffic_light_color', {}).get('color', 'unknown')
                self.traffic_light_counts[color] += 1
                traffic_light_colors.append(color)
        # Track most common color for this frame
        if traffic_light_colors:
            from collections import Counter
            most_common_color = Counter(traffic_light_colors).most_common(1)[0][0]
        else:
            most_common_color = 'unknown'
        now_dt = datetime.now()
        self.traffic_light_color_series.append((now_dt.strftime('%H:%M:%S'), most_common_color))
        self.traffic_light_color_numeric.append(self.traffic_light_color_map.get(most_common_color, 0))
        # Keep last 60 points
        if len(self.traffic_light_color_series) > 60:
            self.traffic_light_color_series = self.traffic_light_color_series[-60:]
            self.traffic_light_color_numeric = self.traffic_light_color_numeric[-60:]
        
        # Update time series data (once per second)
        now = time.time()
        if now - self._last_update >= 1.0:
            self._update_time_series(vehicle_count, pedestrian_count, len(violations), most_common_color)
            self._last_update = now
        
        # Update aggregated metrics
        self._update_aggregated_metrics()
        
        # Emit updated analytics
        self.analytics_updated.emit(self.get_analytics())
    
    def _update_time_series(self, vehicle_count, pedestrian_count, violation_count, traffic_light_color=None):
        """Update time series data for charts"""
        now = datetime.now()
        
        # Check if we've moved to a new hour
        if now.hour != self.current_window.hour or now.day != self.current_window.day:
            # Save current window stats
            self._save_window_stats()
            
            # Reset for new window
            self.current_window = now.replace(minute=0, second=0, microsecond=0)
            self.window_stats = defaultdict(int)
        # Add current counts to window
        self.window_stats['vehicles'] += vehicle_count
        self.window_stats['pedestrians'] += pedestrian_count
        self.window_stats['violations'] += violation_count
        
        # Add to time series
        self.time_series['timestamps'].append(now.strftime('%H:%M:%S'))
        self.time_series['vehicle_counts'].append(vehicle_count)
        self.time_series['pedestrian_counts'].append(pedestrian_count)
        self.time_series['violation_counts'].append(violation_count)
        
        # Add traffic light color to time series
        if traffic_light_color is not None:
            if 'traffic_light_colors' not in self.time_series:
                self.time_series['traffic_light_colors'] = []
            self.time_series['traffic_light_colors'].append(traffic_light_color)
            if len(self.time_series['traffic_light_colors']) > 60:
                self.time_series['traffic_light_colors'] = self.time_series['traffic_light_colors'][-60:]
        
        # Keep last 60 data points (1 minute at 1 Hz)
        if len(self.time_series['timestamps']) > 60:
            for key in self.time_series:
                self.time_series[key] = self.time_series[key][-60:]
    
    def _save_window_stats(self):
        """Save stats for the current time window"""
        if sum(self.window_stats.values()) > 0:
            window_info = {
                'time': self.current_window,
                'vehicles': self.window_stats['vehicles'],
                'pedestrians': self.window_stats['pedestrians'],
                'violations': self.window_stats['violations']
            }
            
            # Update peak stats
            if window_info['vehicles'] > self.aggregated_metrics['peak_vehicle_count']:
                self.aggregated_metrics['peak_vehicle_count'] = window_info['vehicles']
                
            if window_info['violations'] > 0:
                if self.aggregated_metrics['peak_violation_hour'] is None or \
                   window_info['violations'] > self.aggregated_metrics['peak_violation_hour']['violations']:
                    self.aggregated_metrics['peak_violation_hour'] = {
                        'time': self.current_window.strftime('%H:%M'),
                        'violations': window_info['violations']
                    }
    
    def _update_aggregated_metrics(self):
        """Update aggregated analytics metrics"""
        # Count totals
        self.aggregated_metrics['total_vehicles'] = sum([
            self.detection_counts[c] for c in 
            ['car', 'truck', 'bus', 'motorcycle']
        ])
        self.aggregated_metrics['total_pedestrians'] = self.detection_counts['person']
        self.aggregated_metrics['total_violations'] = sum(self.violation_counts.values())
        
        # Average FPS
        if self.fps_history:
            # Only sum numbers, skip dicts
            numeric_fps = [f for f in self.fps_history if isinstance(f, (int, float))]
            if numeric_fps:
                self.aggregated_metrics['avg_fps'] = sum(numeric_fps) / len(numeric_fps)
            else:
                self.aggregated_metrics['avg_fps'] = 0.0
        
        # Average processing time
        if self.processing_times:
            self.aggregated_metrics['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
    
    def get_analytics(self) -> Dict:
        """
        Get current analytics data.
        
        Returns:
            Dictionary of analytics data
        """
        return {
            'detection_counts': dict(self.detection_counts),
            'violation_counts': dict(self.violation_counts),
            'time_series': self.time_series,
            'metrics': self.aggregated_metrics,
            'recent_violations': self.violation_history[-10:] if self.violation_history else [],
            'traffic_light_counts': dict(self.traffic_light_counts),
            'traffic_light_color_series': self.traffic_light_color_series,
            'traffic_light_color_numeric': self.traffic_light_color_numeric
        }
    
    def get_violation_history(self) -> List:
        """
        Get violation history.
        
        Returns:
            List of violation events
        """
        return self.violation_history.copy()
    
    def clear_statistics(self):
        """Reset all statistics"""
        self.detection_counts = defaultdict(int)
        self.violation_counts = defaultdict(int)
        self.detection_history = []
        self.violation_history = []
        self.time_series = {
            'timestamps': [],
            'vehicle_counts': [],
            'pedestrian_counts': [],
            'violation_counts': []
        }
        self.fps_history.clear()
        self.processing_times.clear()
        self.window_stats = defaultdict(int)
        self.aggregated_metrics = {
            'total_vehicles': 0,
            'total_pedestrians': 0,
            'total_violations': 0,
            'avg_processing_time': 0,
            'avg_fps': 0,
            'peak_vehicle_count': 0,
            'peak_violation_hour': None
        }
    
    def register_violation(self, violation):
        """
        Register a new violation in the analytics.
        
        Args:
            violation: Dictionary with violation information
        """
        try:
            # Add to violation counts - check both 'violation' and 'violation_type' keys
            violation_type = violation.get('violation_type') or violation.get('violation', 'unknown')
            self.violation_counts[violation_type] += 1
            
            # Add to violation history
            self.violation_history.append(violation)
            
            # Update time series
            now = datetime.now()
            self.time_series['timestamps'].append(now)
            
            # If we've been running for a while, we might need to drop old timestamps
            if len(self.time_series['timestamps']) > 100:  # Keep last 100 points
                self.time_series['timestamps'] = self.time_series['timestamps'][-100:]
                self.time_series['vehicle_counts'] = self.time_series['vehicle_counts'][-100:]
                self.time_series['pedestrian_counts'] = self.time_series['pedestrian_counts'][-100:]
                self.time_series['violation_counts'] = self.time_series['violation_counts'][-100:]
            
            # Append current totals to time series
            self.time_series['violation_counts'].append(sum(self.violation_counts.values()))
            
            # Make sure all time series have the same length
            while len(self.time_series['vehicle_counts']) < len(self.time_series['timestamps']):
                self.time_series['vehicle_counts'].append(sum(self.detection_counts.get(c, 0) 
                                                           for c in ['car', 'truck', 'bus', 'motorcycle']))
                
            while len(self.time_series['pedestrian_counts']) < len(self.time_series['timestamps']):
                self.time_series['pedestrian_counts'].append(self.detection_counts.get('person', 0))
            
            # Update aggregated metrics
            self.aggregated_metrics['total_violations'] = sum(self.violation_counts.values())
            
            # Emit updated analytics
            self._emit_analytics_update()
            
            print(f"📊 Registered violation in analytics: {violation_type}")
        except Exception as e:
            print(f"❌ Error registering violation in analytics: {e}")
            import traceback
            traceback.print_exc()

    def _emit_analytics_update(self):
        """Emit analytics update signal with current data"""
        try:
            self.analytics_updated.emit(self.get_analytics())
        except Exception as e:
            print(f"❌ Error emitting analytics update: {e}")
            import traceback
            traceback.print_exc()
