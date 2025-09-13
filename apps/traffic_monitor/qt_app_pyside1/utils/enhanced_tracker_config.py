"""
Enhanced Tracker Configuration and Usage Examples

This file demonstrates how to configure and use the enhanced tracker
with real intersection geometry and movement constraints.
"""

import numpy as np
from utils.enhanced_tracker import EnhancedVehicleTracker


class IntersectionConfig:
    """Configuration for smart intersection tracking"""
    
    def __init__(self):
        self.tracker = None
        self.roi_configs = []
        
    def setup_four_way_intersection(self, frame_width=1920, frame_height=1080):
        """
        Setup ROI and movement constraints for a typical 4-way intersection
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        # Initialize enhanced tracker
        self.tracker = EnhancedVehicleTracker()
        
        # Define intersection lanes with expected movement directions
        # Coordinates are example values - adjust for your specific intersection
        
        # North-South lanes (vertical movement)
        north_lane = [
            (int(frame_width * 0.4), 0),                    # Top left
            (int(frame_width * 0.6), 0),                    # Top right  
            (int(frame_width * 0.6), int(frame_height * 0.4)), # Bottom right
            (int(frame_width * 0.4), int(frame_height * 0.4))  # Bottom left
        ]
        
        south_lane = [
            (int(frame_width * 0.4), int(frame_height * 0.6)), # Top left
            (int(frame_width * 0.6), int(frame_height * 0.6)), # Top right
            (int(frame_width * 0.6), frame_height),             # Bottom right
            (int(frame_width * 0.4), frame_height)              # Bottom left
        ]
        
        # East-West lanes (horizontal movement)
        east_lane = [
            (0, int(frame_height * 0.4)),                   # Top left
            (int(frame_width * 0.4), int(frame_height * 0.4)), # Top right
            (int(frame_width * 0.4), int(frame_height * 0.6)), # Bottom right
            (0, int(frame_height * 0.6))                    # Bottom left
        ]
        
        west_lane = [
            (int(frame_width * 0.6), int(frame_height * 0.4)), # Top left
            (frame_width, int(frame_height * 0.4)),             # Top right
            (frame_width, int(frame_height * 0.6)),             # Bottom right
            (int(frame_width * 0.6), int(frame_height * 0.6))  # Bottom left
        ]
        
        # Crosswalk areas (pedestrian zones)
        north_crosswalk = [
            (int(frame_width * 0.35), int(frame_height * 0.35)),
            (int(frame_width * 0.65), int(frame_height * 0.35)),
            (int(frame_width * 0.65), int(frame_height * 0.4)),
            (int(frame_width * 0.35), int(frame_height * 0.4))
        ]
        
        south_crosswalk = [
            (int(frame_width * 0.35), int(frame_height * 0.6)),
            (int(frame_width * 0.65), int(frame_height * 0.6)),
            (int(frame_width * 0.65), int(frame_height * 0.65)),
            (int(frame_width * 0.35), int(frame_height * 0.65))
        ]
        
        # Add ROIs with movement constraints
        self.tracker.add_intersection_roi(north_lane, expected_direction=[0, 1])   # Southbound
        self.tracker.add_intersection_roi(south_lane, expected_direction=[0, -1])  # Northbound
        self.tracker.add_intersection_roi(east_lane, expected_direction=[1, 0])    # Eastbound
        self.tracker.add_intersection_roi(west_lane, expected_direction=[-1, 0])   # Westbound
        
        # Crosswalks without strict direction constraints (pedestrians can move freely)
        self.tracker.add_intersection_roi(north_crosswalk, expected_direction=None)
        self.tracker.add_intersection_roi(south_crosswalk, expected_direction=None)
        
        print("[INTERSECTION_CONFIG] 4-way intersection setup complete")
        print(f"[INTERSECTION_CONFIG] Configured for {frame_width}x{frame_height} resolution")
        
        return self.tracker
    
    def setup_highway_onramp(self, frame_width=1920, frame_height=1080):
        """
        Setup ROI for highway on-ramp monitoring
        """
        self.tracker = EnhancedVehicleTracker()
        
        # Main highway lanes (left to right movement)
        highway_lanes = [
            (0, int(frame_height * 0.3)),
            (frame_width, int(frame_height * 0.3)),
            (frame_width, int(frame_height * 0.7)),
            (0, int(frame_height * 0.7))
        ]
        
        # On-ramp merging area (curved movement)
        onramp_area = [
            (int(frame_width * 0.1), int(frame_height * 0.7)),
            (int(frame_width * 0.4), int(frame_height * 0.6)),
            (int(frame_width * 0.6), int(frame_height * 0.5)),
            (int(frame_width * 0.2), int(frame_height * 0.8))
        ]
        
        # Add ROIs
        self.tracker.add_intersection_roi(highway_lanes, expected_direction=[1, 0])  # Eastbound
        self.tracker.add_intersection_roi(onramp_area, expected_direction=[0.8, -0.6])  # Merging direction
        
        print("[INTERSECTION_CONFIG] Highway on-ramp setup complete")
        return self.tracker


class TrackerPerformanceMonitor:
    """Monitor tracking performance and provide analytics"""
    
    def __init__(self):
        self.frame_count = 0
        self.track_stats = {}
        self.id_switches = 0
        self.new_tracks_created = 0
        
    def update_stats(self, tracks):
        """Update tracking statistics"""
        self.frame_count += 1
        current_ids = {track['id'] for track in tracks}
        
        # Count new tracks
        for track in tracks:
            track_id = track['id']
            if track_id not in self.track_stats:
                self.track_stats[track_id] = {
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'total_frames': 1,
                    'avg_confidence': track['confidence'],
                    'max_velocity': np.linalg.norm(track.get('velocity', [0, 0]))
                }
                self.new_tracks_created += 1
            else:
                # Update existing track stats
                stats = self.track_stats[track_id]
                stats['last_seen'] = self.frame_count
                stats['total_frames'] += 1
                stats['avg_confidence'] = (stats['avg_confidence'] + track['confidence']) / 2
                velocity_norm = np.linalg.norm(track.get('velocity', [0, 0]))
                stats['max_velocity'] = max(stats['max_velocity'], velocity_norm)
    
    def get_performance_summary(self):
        """Get tracking performance summary"""
        if not self.track_stats:
            return "No tracking data available"
        
        total_tracks = len(self.track_stats)
        avg_track_length = np.mean([stats['total_frames'] for stats in self.track_stats.values()])
        avg_confidence = np.mean([stats['avg_confidence'] for stats in self.track_stats.values()])
        
        summary = f"""
=== Enhanced Tracker Performance Summary ===
Total Frames Processed: {self.frame_count}
Total Unique Tracks: {total_tracks}
Average Track Length: {avg_track_length:.1f} frames
Average Confidence: {avg_confidence:.3f}
New Tracks Created: {self.new_tracks_created}
Tracks/Frame Ratio: {total_tracks / max(1, self.frame_count):.3f}
"""
        return summary


# Usage Example
def example_usage():
    """Example of how to use the enhanced tracker"""
    
    # Setup intersection configuration
    config = IntersectionConfig()
    tracker = config.setup_four_way_intersection(frame_width=1920, frame_height=1080)
    
    # Setup performance monitoring
    monitor = TrackerPerformanceMonitor()
    
    # Example detection processing loop
    def process_frame(detections, frame=None):
        """Process a single frame with detections"""
        
        # Update tracker
        tracks = tracker.update(detections, frame)
        
        # Update performance stats
        monitor.update_stats(tracks)
        
        # Log enhanced tracking info
        for track in tracks:
            velocity = track.get('velocity', [0, 0])
            speed = np.linalg.norm(velocity)
            
            print(f"Track ID: {track['id']}, "
                  f"Confidence: {track['confidence']:.3f}, "
                  f"Speed: {speed:.1f} px/frame, "
                  f"Age: {track['age']} frames, "
                  f"Stable: {track['stable_frames']} frames")
        
        return tracks
    
    # Example detections (replace with real detector output)
    example_detections = [
        {
            'bbox': [100, 200, 200, 300],  # [x1, y1, x2, y2]
            'confidence': 0.8,
            'class_id': 0
        },
        {
            'bbox': [300, 400, 400, 500],
            'confidence': 0.7,
            'class_id': 0
        }
    ]
    
    # Process example frame
    tracks = process_frame(example_detections)
    
    # Print performance summary
    print(monitor.get_performance_summary())
    
    return tracker, monitor


if __name__ == "__main__":
    # Run example
    tracker, monitor = example_usage()
