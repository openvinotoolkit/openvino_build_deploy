#!/usr/bin/env python3
"""
Smart Intersection Services Setup Script
Automated setup for MQTT + InfluxDB + Grafana integration
"""

import os
import sys
import json
import time
import zipfile
import subprocess
import requests
from pathlib import Path
from urllib.request import urlretrieve

class ServicesSetup:
    """Automated setup for Smart Intersection services"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.services_dir = self.base_dir / "services"
        self.downloads_dir = self.services_dir / "downloads"
        self.install_dir = Path("C:/SmartIntersectionServices")
        
        # Service configurations
        self.services = {
            "mosquitto": {
                "name": "Eclipse Mosquitto MQTT Broker",
                "url": "https://mosquitto.org/files/binary/win64/mosquitto-2.0.18-install-windows-x64.exe",
                "filename": "mosquitto-2.0.18-install-windows-x64.exe",
                "install_dir": "C:/Program Files/mosquitto",
                "port": 1883,
                "check_url": None
            },
            "influxdb": {
                "name": "InfluxDB v2",
                "url": "https://github.com/influxdata/influxdb/releases/download/v2.7.11/influxdb2-2.7.11-windows-amd64.zip",
                "filename": "influxdb2-2.7.11-windows-amd64.zip",
                "install_dir": Path("C:/Users/devcloud/Desktop/Clean/clean-final-push/qt_app_pyside1/services/services/downloads/influxdb2-2.7.11-windows"),
                "port": 8086,
                "check_url": "http://localhost:8086/health"
            },
            "grafana": {
                "name": "Grafana",
                "url": "https://dl.grafana.com/oss/release/grafana-10.2.2.windows-amd64.zip",
                "filename": "grafana-10.2.2.windows-amd64.zip", 
                "install_dir": Path("C:/Users/devcloud/Desktop/Clean/clean-final-push/qt_app_pyside1/services/services/downloads/grafana-10.2.2.windows-amd64"),
                "port": 3000,
                "check_url": "http://localhost:3000/api/health"
            }
        }
    
    def run_setup(self):
        """Run the complete setup process"""
        print("=" * 70)
        print("🚦 SMART INTERSECTION SERVICES SETUP")
        print("=" * 70)
        print()
        
        # Check if running as administrator (for Mosquitto installation)
        if not self._is_admin():
            print("⚠️  Administrator privileges recommended for complete setup")
            print("   Some services may require manual installation")
            print()
        
        # Create directories
        self._create_directories()
        
        # Check Docker availability
        docker_available = self._check_docker()
        
        if docker_available:
            print("✅ Docker is available!")
            print("   Recommended: Use Docker Compose for easy setup")
            print("   Alternative: Manual installation")
            print()
            
            choice = input("Use Docker Compose setup? (y/n) [y]: ").strip().lower()
            if choice in ['', 'y', 'yes']:
                self._setup_docker_compose()
                return
        
        print("🔧 Proceeding with manual installation...")
        print()
        
        # Install Python dependencies
        self._install_python_dependencies()
        
        # Setup each service
        for service_key, service_config in self.services.items():
            print(f"📦 Setting up {service_config['name']}...")
            self._setup_service(service_key, service_config)
            print()
        
        # Configure services
        self._configure_services()
        
        # Test services
        self._test_services()
        
        # Create startup scripts
        self._create_startup_scripts()
        
        # Final instructions
        self._show_final_instructions()
    
    def _is_admin(self):
        """Check if running with administrator privileges"""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def _create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directories...")
        
        directories = [
            self.downloads_dir,
            self.install_dir,
            self.install_dir / "influxdb" / "data",
            self.install_dir / "grafana" / "data",
            self.services_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        print()
    
    def _check_docker(self):
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def _install_python_dependencies(self):
        """Install required Python packages"""
        print("🐍 Installing Python dependencies...")
        
        packages = [
            "paho-mqtt==1.6.1",
            "influxdb-client==1.43.0", 
            "grafana-api==1.0.3",
            "aiohttp==3.9.1",
            "asyncio-mqtt==0.13.0",
            "requests>=2.28.0"
        ]
        
        for package in packages:
            try:
                print(f"   Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"   ✅ {package}")
                else:
                    print(f"   ⚠️ {package} - {result.stderr.strip()}")
            except Exception as e:
                print(f"   ❌ Failed to install {package}: {e}")
        print()
    
    def _setup_service(self, service_key: str, config: dict):
        """Setup individual service"""
        filename = config["filename"]
        download_path = self.downloads_dir / filename
        
        # Download if not exists
        if not download_path.exists():
            print(f"   📥 Downloading {config['name']}...")
            try:
                self._download_with_progress(config["url"], download_path)
                print(f"   ✅ Downloaded to {download_path}")
            except Exception as e:
                print(f"   ❌ Download failed: {e}")
                print(f"   📌 Manual download: {config['url']}")
                return
        else:
            print(f"   ✅ Already downloaded: {filename}")
        
        # Install/Extract service
        if service_key == "mosquitto":
            self._install_mosquitto(download_path)
        elif filename.endswith(".zip"):
            self._extract_service(download_path, config["install_dir"])
    
    def _download_with_progress(self, url: str, filepath: Path):
        """Download file with progress indicator"""
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\r   Progress: {percent}%", end="", flush=True)
        
        urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
    
    def _install_mosquitto(self, installer_path: Path):
        """Install Mosquitto MQTT broker"""
        print(f"   🔧 Installing Mosquitto...")
        print(f"   📌 Please run as administrator: {installer_path}")
        print(f"   📌 Install to default location: C:\\Program Files\\mosquitto\\")
        
        # Attempt automatic installation if admin
        if self._is_admin():
            try:
                result = subprocess.run(
                    [str(installer_path), "/S"],  # Silent install
                    timeout=300
                )
                if result.returncode == 0:
                    print(f"   ✅ Mosquitto installed successfully")
                else:
                    print(f"   ⚠️ Installation may require manual confirmation")
            except Exception as e:
                print(f"   ⚠️ Automatic installation failed: {e}")
                print(f"   📌 Please run manually: {installer_path}")
        else:
            print(f"   📌 Run manually as administrator: {installer_path}")
    
    def _extract_service(self, zip_path: Path, install_dir: Path):
        """Extract service from zip file"""
        print(f"   📦 Extracting to {install_dir}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to temporary directory first
                temp_dir = install_dir.parent / f"temp_{install_dir.name}"
                zip_ref.extractall(temp_dir)
                
                # Find the main directory in extracted files
                extracted_items = list(temp_dir.iterdir())
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    # Move contents from subdirectory
                    source_dir = extracted_items[0]
                    install_dir.mkdir(parents=True, exist_ok=True)
                    
                    for item in source_dir.iterdir():
                        target = install_dir / item.name
                        if item.is_dir():
                            import shutil
                            shutil.copytree(item, target, dirs_exist_ok=True)
                        else:
                            import shutil
                            shutil.copy2(item, target)
                    
                    # Clean up temporary directory
                    import shutil
                    shutil.rmtree(temp_dir)
                else:
                    # Move entire temp directory
                    import shutil
                    if install_dir.exists():
                        shutil.rmtree(install_dir)
                    shutil.move(temp_dir, install_dir)
                
                print(f"   ✅ Extracted successfully")
                
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
    
    def _configure_services(self):
        """Configure all services"""
        print("⚙️ Configuring services...")
        
        # Copy configuration files
        config_mappings = [
            (self.base_dir / "mqtt" / "mosquitto.conf", 
             Path("C:/Program Files/mosquitto/mosquitto.conf")),
            (self.base_dir / "influxdb" / "config.yml",
             Path("C:/Users/devcloud/Desktop/Clean/clean-final-push/qt_app_pyside1/services/services/downloads/influxdb2-2.7.11-windows/config.yml")),
            (self.base_dir / "grafana" / "datasources" / "influxdb.yml",
             Path("C:/Users/devcloud/Desktop/Clean/clean-final-push/qt_app_pyside1/services/services/downloads/grafana-10.2.2.windows-amd64/conf/provisioning/datasources/influxdb.yml"))
        ]
        
        for source, target in config_mappings:
            if source.exists():
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(source, target)
                    print(f"   ✅ Configured {target.name}")
                except Exception as e:
                    print(f"   ⚠️ Could not copy {source.name}: {e}")
            else:
                print(f"   ⚠️ Source config not found: {source}")
        print()
    
    def _setup_docker_compose(self):
        """Setup using Docker Compose"""
        print("🐳 Setting up with Docker Compose...")
        
        compose_file = self.services_dir / "docker" / "docker-compose.yml"
        if not compose_file.exists():
            print("❌ Docker Compose file not found!")
            return
        
        try:
            # Change to docker directory
            docker_dir = self.services_dir / "docker"
            os.chdir(docker_dir)
            
            # Pull images
            print("   📥 Pulling Docker images...")
            subprocess.run(["docker-compose", "pull"], check=True)
            
            # Start services
            print("   🚀 Starting services...")
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            
            print("   ✅ Docker services started successfully!")
            print()
            
            # Wait for services to be ready
            print("   ⏳ Waiting for services to initialize...")
            time.sleep(30)
            
            # Test services
            self._test_docker_services()
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Docker Compose failed: {e}")
        except Exception as e:
            print(f"   ❌ Setup failed: {e}")
    
    def _test_services(self):
        """Test if services are running"""
        print("🧪 Testing services...")
        
        for service_key, config in self.services.items():
            port = config["port"]
            check_url = config.get("check_url")
            
            print(f"   Testing {config['name']} (port {port})...")
            
            # Test port
            if self._test_port(port):
                print(f"   ✅ Port {port} is open")
                
                # Test HTTP endpoint if available
                if check_url:
                    if self._test_http(check_url):
                        print(f"   ✅ HTTP endpoint responding")
                    else:
                        print(f"   ⚠️ HTTP endpoint not ready yet")
            else:
                print(f"   ❌ Port {port} not responding")
                print(f"   📌 Make sure {config['name']} is running")
        print()
    
    def _test_docker_services(self):
        """Test Docker services"""
        print("   🧪 Testing Docker services...")
        
        services_to_test = [
            ("MQTT Broker", 1883, None),
            ("InfluxDB", 8086, "http://localhost:8086/health"),
            ("Grafana", 3000, "http://localhost:3000/api/health")
        ]
        
        for name, port, url in services_to_test:
            print(f"     Testing {name}...")
            
            if self._test_port(port):
                print(f"     ✅ {name} port {port} is open")
                
                if url and self._test_http(url):
                    print(f"     ✅ {name} HTTP endpoint responding")
            else:
                print(f"     ❌ {name} port {port} not responding")
    
    def _test_port(self, port: int) -> bool:
        """Test if port is open"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _test_http(self, url: str) -> bool:
        """Test HTTP endpoint"""
        try:
            response = requests.get(url, timeout=10)
            return response.status_code in [200, 204]
        except:
            return False
    
    def _create_startup_scripts(self):
        """Create startup scripts"""
        print("📝 Creating startup scripts...")
        
        # Scripts are already created in the setup
        scripts = [
            self.services_dir / "scripts" / "start_services.bat",
            self.services_dir / "scripts" / "stop_services.bat"
        ]
        
        for script in scripts:
            if script.exists():
                print(f"   ✅ {script.name}")
            else:
                print(f"   ⚠️ {script.name} not found")
        print()
    
    def _show_final_instructions(self):
        """Show final setup instructions"""
        print("🎉 SETUP COMPLETE!")
        print("=" * 50)
        print()
        print("📋 Next Steps:")
        print("1. Start services:")
        print(f"   Run: {self.services_dir}/scripts/start_services.bat")
        print()
        print("2. Access services:")
        print("   🌐 Grafana: http://localhost:3000 (admin/admin)")
        print("   📊 InfluxDB: http://localhost:8086")
        print("   📡 MQTT: localhost:1883")
        print()
        print("3. Run desktop application:")
        print("   cd qt_app_pyside1")
        print("   python main.py")
        print()
        print("4. Service status:")
        print("   Desktop app will show service connection status")
        print("   Green indicators = services connected")
        print()
        print("🔧 Configuration:")
        print(f"   Services config: {self.base_dir}/config/smart-intersection/services-config.json")
        print(f"   MQTT topics: {self.services_dir}/mqtt/topics.json")
        print()
        print("📊 Grafana Dashboards:")
        print("   Pre-configured dashboards will load automatically")
        print("   View real-time traffic analytics and performance metrics")
        print()
        print("💡 Tips:")
        print("   - Services may take 30-60 seconds to fully initialize")
        print("   - Check Windows Firewall if connections fail")
        print("   - Use Docker Compose for easier management")
        print("   - Monitor logs in services/logs/ directory")
        print()


def main():
    """Main setup function"""
    try:
        setup = ServicesSetup()
        setup.run_setup()
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
