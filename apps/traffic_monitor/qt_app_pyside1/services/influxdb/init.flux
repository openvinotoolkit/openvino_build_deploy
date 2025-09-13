// Smart Intersection InfluxDB Initialization Script
// This Flux script sets up the initial database structure

// Create bucket for traffic monitoring data
option task = {name: "smart-intersection-init", every: 1mo}

// Import required packages
import "influxdata/influxdb/schema"

// Create retention policy for traffic data (7 days)
buckets.create(
    bucket: "traffic_monitoring",
    orgID: "smart-intersection-org", 
    retentionPeriod: 168h,
    description: "Smart intersection traffic monitoring data"
)

// Create bucket for long-term analytics (30 days)
buckets.create(
    bucket: "traffic_analytics",
    orgID: "smart-intersection-org",
    retentionPeriod: 720h,
    description: "Long-term traffic analytics and trends"
)

// Create measurements schema

// Performance metrics measurement
schema.create(
    bucket: "traffic_monitoring",
    measurement: "performance",
    columns: {
        timestamp: "timestamp",
        fps: "float",
        gpu_usage: "float", 
        memory_usage: "float",
        processing_time_ms: "float",
        camera_count: "int",
        active_objects: "int"
    },
    tags: ["service", "camera_id"]
)

// Detection events measurement  
schema.create(
    bucket: "traffic_monitoring",
    measurement: "detection_events",
    columns: {
        timestamp: "timestamp",
        object_count: "int",
        vehicle_count: "int", 
        pedestrian_count: "int",
        confidence_avg: "float",
        processing_time: "float"
    },
    tags: ["camera_id", "camera_position", "detection_model"]
)

// Violation events measurement
schema.create(
    bucket: "traffic_monitoring", 
    measurement: "violation_events",
    columns: {
        timestamp: "timestamp",
        violation_type: "string",
        severity_level: "string",
        object_id: "string",
        confidence: "float"
    },
    tags: ["camera_id", "roi_id", "violation_category"]
)

// Traffic flow measurement
schema.create(
    bucket: "traffic_monitoring",
    measurement: "traffic_flow",
    columns: {
        timestamp: "timestamp",
        vehicle_count: "int",
        average_speed: "float",
        lane_occupancy: "float",
        congestion_level: "string"
    },
    tags: ["lane_id", "direction", "camera_position"]
)

// ROI events measurement
schema.create(
    bucket: "traffic_monitoring",
    measurement: "roi_events", 
    columns: {
        timestamp: "timestamp",
        event_type: "string",
        dwell_time: "float",
        object_count: "int",
        activity_level: "float"
    },
    tags: ["roi_id", "roi_type", "camera_id"]
)

// System health measurement
schema.create(
    bucket: "traffic_monitoring",
    measurement: "system_health",
    columns: {
        timestamp: "timestamp", 
        cpu_usage: "float",
        memory_usage: "float",
        disk_usage: "float",
        service_status: "string",
        uptime_seconds: "int"
    },
    tags: ["service_name", "host"]
)

// Create continuous queries for analytics

// Average FPS over 1 minute intervals
option task = {name: "fps_analytics", every: 1m}

from(bucket: "traffic_monitoring")
    |> range(start: -1m)
    |> filter(fn: (r) => r._measurement == "performance")
    |> filter(fn: (r) => r._field == "fps")
    |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
    |> set(key: "_measurement", value: "fps_avg_1m")
    |> to(bucket: "traffic_analytics", org: "smart-intersection-org")

// Vehicle count aggregation over 5 minute intervals  
option task = {name: "vehicle_count_analytics", every: 5m}

from(bucket: "traffic_monitoring")
    |> range(start: -5m)
    |> filter(fn: (r) => r._measurement == "detection_events")
    |> filter(fn: (r) => r._field == "vehicle_count")
    |> aggregateWindow(every: 5m, fn: sum, createEmpty: false)
    |> set(key: "_measurement", value: "vehicle_count_5m")
    |> to(bucket: "traffic_analytics", org: "smart-intersection-org")

// Violation count by type over 1 hour
option task = {name: "violation_analytics", every: 1h}

from(bucket: "traffic_monitoring")
    |> range(start: -1h)
    |> filter(fn: (r) => r._measurement == "violation_events")
    |> group(columns: ["violation_type"])
    |> count()
    |> set(key: "_measurement", value: "violation_count_1h") 
    |> to(bucket: "traffic_analytics", org: "smart-intersection-org")
