import threading
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import queue
import time

@dataclass
class GestureEvent:
    """Represents a gesture-related event"""
    event_type: str 
    timestamp: datetime
    data: Dict[str, Any]
    source: str = 'engine'

class EventBus:
    """Thread-safe event system for gesture engine communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._event_queue = queue.Queue()
        self._processing = False
        self._processor_thread = None
    
    def subscribe(self, event_type: str, callback: Callable[[GestureEvent], None]):
        """Subscribe to specific event types"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
    
    def publish(self, event: GestureEvent):
        """Publish an event (thread-safe)"""
        self._event_queue.put(event)
    
    def start_processing(self):
        """Start event processing in background thread"""
        if not self._processing:
            self._processing = True
            self._processor_thread = threading.Thread(target=self._process_events, daemon=True)
            self._processor_thread.start()
    
    def stop_processing(self):
        """Stop event processing"""
        self._processing = False
        if self._processor_thread:
            self._processor_thread.join(timeout=1.0)
    
    def _process_events(self):
        """Process events in background thread"""
        while self._processing:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def _dispatch_event(self, event: GestureEvent):
        """Dispatch event to subscribers"""
        with self._lock:
            
            if event.event_type in self._subscribers:
                for callback in self._subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"Error in event callback: {e}")
            
            
            if '*' in self._subscribers:
                for callback in self._subscribers['*']:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"Error in wildcard callback: {e}")


event_bus = EventBus()