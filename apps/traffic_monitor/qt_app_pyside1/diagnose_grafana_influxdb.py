"""
Comprehensive diagnostic script for InfluxDB and Grafana integration
"""

import requests
import json
import time
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point

# Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "kNFfXEpPQoWrk5Tteowda21Dzv6xD3jY7QHSHHQHb5oYW6VH6mkAgX9ZMjQJkaHHa8FwzmyVFqDG7qqzxN09uQ=="
INFLUX_ORG = "smart-intersection-org"
INFLUX_BUCKET = "traffic_monitoring"
GRAFANA_URL = "http://localhost:3000"

def check_influxdb():
    """Check InfluxDB connection and data"""
    print("\n===== CHECKING INFLUXDB =====")
    try:
        # Connect to InfluxDB
        print(f"Connecting to InfluxDB at {INFLUX_URL}...")
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        
        # Check health
        health = client.health()
        print(f"InfluxDB health: {health.status}")
        
        # Check buckets
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        bucket_names = [b.name for b in buckets]
        
        print(f"Found {len(buckets)} buckets: {', '.join(bucket_names)}")
        
        if INFLUX_BUCKET in bucket_names:
            print(f"✅ '{INFLUX_BUCKET}' bucket exists!")
        else:
            print(f"❌ '{INFLUX_BUCKET}' bucket NOT found!")
            return False
        
        # Check recent data
        query_api = client.query_api()
        
        # Query performance data from the last 15 minutes
        print("\nChecking for recent performance data (last 15 minutes)...")
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
          |> range(start: -15m)
          |> filter(fn: (r) => r._measurement == "performance")
          |> count()
        '''
        
        result = query_api.query(query)
        
        if not result or len(result) == 0:
            print("❌ No performance data found in the last 15 minutes!")
            
            # Write a test point
            print("\nWriting a test data point to InfluxDB...")
            write_api = client.write_api()
            point = Point("performance") \
                .tag("camera_id", "diagnostic_test") \
                .field("fps", 30.0) \
                .field("processing_time_ms", 25.0) \
                .field("gpu_usage", 45.0) \
                .time(datetime.utcnow())
            
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
            print("Test point written successfully.")
        else:
            count = 0
            for table in result:
                for record in table.records:
                    count = record.get_value()
            
            print(f"✅ Found {count} performance data points in the last 15 minutes!")
        
        print("\nChecking most recent data in each measurement...")
        for measurement in ["performance", "detection_events", "violation_events"]:
            query = f'''
            from(bucket: "{INFLUX_BUCKET}")
              |> range(start: -15m)
              |> filter(fn: (r) => r._measurement == "{measurement}")
              |> group()
              |> last()
            '''
            
            result = query_api.query(query)
            if not result or len(list(result)) == 0:
                print(f"❌ No recent data in '{measurement}' measurement")
            else:
                print(f"✅ Found recent data in '{measurement}' measurement")
                for table in result:
                    for record in table.records:
                        print(f"   - {record.get_field()}: {record.get_value()}")
        
        return True
        
    except Exception as e:
        print(f"❌ InfluxDB Error: {e}")
        return False

def check_grafana_datasource():
    """Check Grafana datasource configuration"""
    print("\n===== CHECKING GRAFANA =====")
    try:
        # For proper auth, you'd need admin credentials
        # This is just a basic connectivity check
        print(f"Checking Grafana server at {GRAFANA_URL}...")
        health_resp = requests.get(f"{GRAFANA_URL}/api/health")
        
        if health_resp.status_code != 200:
            print(f"❌ Grafana is not responding correctly: {health_resp.status_code}")
            return False
            
        health_data = health_resp.json()
        print(f"Grafana health: {health_data.get('status', 'unknown')}, version: {health_data.get('version', 'unknown')}")
        
        # To check datasources, you'd need auth:
        # For simplicity, we'll print instructions for manual verification
        print("\n🔍 MANUAL GRAFANA VERIFICATION NEEDED:")
        print("1. Open your browser and go to Grafana: http://localhost:3000")
        print("2. Log in with your credentials")
        print("3. Go to Configuration > Data Sources")
        print("4. Click on your InfluxDB data source")
        print("5. Verify these settings:")
        print(f"   - URL: {INFLUX_URL}")
        print(f"   - Organization: {INFLUX_ORG}")
        print("   - Token: [Your token should be set]")
        print(f"   - Default bucket: {INFLUX_BUCKET}")
        print("   - Query language: Flux")
        print("\n6. Click 'Save & Test' - you should see 'Data source is working'")
        print("\n7. Go to your dashboard and:")
        print("   - Check if the time range includes the last 15 minutes")
        print("   - Click the refresh button")
        print("   - If still no data, try setting a wider time range (e.g., last 1 hour)")
        
        return True
        
    except Exception as e:
        print(f"❌ Grafana Check Error: {e}")
        return False

if __name__ == "__main__":
    influx_ok = check_influxdb()
    grafana_ok = check_grafana_datasource()
    
    print("\n===== DIAGNOSIS SUMMARY =====")
    print(f"InfluxDB: {'✅ OK' if influx_ok else '❌ ISSUES DETECTED'}")
    print(f"Grafana: {'✅ OK' if grafana_ok else '❌ ISSUES DETECTED'}")
    
    if influx_ok and grafana_ok:
        print("\n✅ All systems appear to be working correctly.")
        print("If Grafana is still not showing data, please follow the manual verification steps above.")
        print("Also, ensure your dashboard panel queries match the measurement and field names in InfluxDB.")
    else:
        print("\n❌ Issues were detected. Please review the details above.")
