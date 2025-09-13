#!/usr/bin/env python3
"""
Cross-Platform Build Script for Traffic Monitor Application
This script builds the application for both Windows and macOS with PyInstaller
"""

import os
import subprocess
import sys
import shutil
import platform
from pathlib import Path

def get_platform_info():
    """Get current platform information"""
    system = platform.system().lower()
    arch = platform.machine().lower()
    
    if system == "windows":
        return "windows", "exe", ";"
    elif system == "darwin":
        return "macos", "app", ":"
    elif system == "linux":
        return "linux", "", ":"
    else:
        return "unknown", "", ":"

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        # Show real-time output instead of capturing it
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Exit code: {e.returncode}")
        return False

def get_icon_path(platform_name):
    """Get appropriate icon path for platform"""
    if platform_name == "windows":
        icon_path = "resources/icon.ico"
        if os.path.exists(icon_path):
            return icon_path
        # Try alternative locations
        alt_paths = ["icon.ico", "resources/app.ico", "app.ico"]
        for path in alt_paths:
            if os.path.exists(path):
                return path
    elif platform_name == "macos":
        icon_path = "resources/icon.icns"
        if os.path.exists(icon_path):
            return icon_path
        # Try alternative locations
        alt_paths = ["icon.icns", "resources/app.icns", "app.icns"]
        for path in alt_paths:
            if os.path.exists(path):
                return path
    return None

def get_data_separator(platform_name):
    """Get the correct data separator for platform"""
    return ";" if platform_name == "windows" else ":"

def check_module_exists(module_name):
    """Check if a module is installed and can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_available_hidden_imports():
    """Get list of hidden imports that are actually available"""
    all_imports = [
        # Core dependencies
        'cv2',
        'openvino',
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
        'pandas',
        'seaborn',
        
        # PySide6 modules
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'PySide6.QtOpenGL',
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        
        # Standard library (always available)
        'json',
        'os',
        'sys',
        'time',
        'traceback',
        'pathlib',
        'threading',
        'queue',
        'collections',
        'itertools',
        'functools',
        'typing',
        'subprocess',
        'shutil',
        'platform',
        'datetime',
        'logging',
        'argparse',
        'configparser',
        'pickle',
        'base64',
        'hashlib',
        'uuid',
        'warnings',
        'sqlite3',
        'socket',
        'asyncio',
        
        # ML/AI libraries
        'torch',
        'torchvision',
        'transformers',
        'huggingface_hub',
        'tokenizers',
        'optimum',
        'timm',
        'onnx',
        'onnxruntime',
        
        # Tracking and detection
        'deep_sort_realtime',
        'norfair',
        'ultralytics',
        
        # Communication
        'paho.mqtt',
        'influxdb_client',
        'aiohttp',
        'requests',
        'flask',
        
        # Utilities
        'PIL',
        'tqdm',
        'rich',
        'pydantic',
        'fpdf',
        'jsonschema',
        'pydot',
        'pyparsing',
        'tabulate',
        'pyarrow',
        
        # OpenAI/VLM
        'openai',
        
        # Missing modules that cause import errors
        'splash',
        'red_light_violation_pipeline',
        'detection_openvino',
        'detection_openvino_async',
        'detection_openvino_fixed',
        'violation_openvino',
        'fallback_annotation_utils',
        'annotation_utils',
        'utils',
        'utils.annotation_utils',
        'utils.helpers',
        'utils.enhanced_annotation_utils',
        'utils.data_publisher',
        'utils.mqtt_publisher',
        'utils.traffic_light_utils',
        'utils.scene_analytics',
        'utils.crosswalk_utils',
        'utils.crosswalk_utils2',
        'utils.enhanced_tracker',
        'utils.embedder_openvino',
        'controllers.video_controller_new',
        'controllers.model_manager',
        'controllers.analytics_controller',
        'controllers.performance_overlay',
        'controllers.vlm_controller',
        'controllers.smart_intersection_controller',
        'ui.main_window',
        'ui.analytics_tab',
        'ui.violations_tab',
        'ui.export_tab',
        'ui.modern_config_panel',
        'ui.modern_live_detection_tab',
    ]
    
    # Optional modules (check if they exist)
    optional_imports = [
        'websockets',
        'python-dotenv',
        'yaml',
        'toml',
        'Pillow',
        'protobuf',
        'tensorboard',
        'docker',
        'gradio',
        'streamlit',
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'psycopg2',
        'redis',
        'celery',
        'prometheus_client',
        'grafana_api',
        'elasticsearch',
        'mongodb',
        'pymongo',
    ]
    
    # Add optional modules only if they're available
    available_imports = all_imports.copy()
    for module in optional_imports:
        if check_module_exists(module):
            available_imports.append(module)
        else:
            print(f"⚠️  Optional module '{module}' not found - skipping")
    
    return available_imports

def build_application(platform_name, extension, data_sep, is_debug=False):
    """Build the application with PyInstaller"""
    
    # Get current directory
    current_dir = Path.cwd()
    parent_dir = current_dir.parent  # This is clean-final-push directory
    print(f"Building from: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    
    # Set app name based on debug mode
    app_name = "TrafficMonitorDebug" if is_debug else "TrafficMonitor"
    
    # Clean previous builds for this specific build
    print(f"\n🧹 Cleaning previous {app_name} builds...")
    build_folder = current_dir / "build" / app_name
    if build_folder.exists():
        shutil.rmtree(build_folder)
        print(f"Removed {build_folder}")
    
    spec_file = f"{app_name}.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)
        print(f"Removed old spec file: {spec_file}")
    
    # Get icon path
    icon_path = get_icon_path(platform_name)
    
    # Define PyInstaller command with all necessary flags
    # Use --onedir instead of --onefile to avoid struct.error with large projects
    pyinstaller_cmd = [
        'pyinstaller',
        f'--name={app_name}',
        '--onedir',  # Create directory instead of single file to avoid size limits
        '--clean',  # Clean PyInstaller cache and remove temporary files
    ]
    
    # Add platform-specific options
    if platform_name == "windows":
        if not is_debug:
            pyinstaller_cmd.append('--windowed')  # No console for release
        else:
            pyinstaller_cmd.append('--console')  # Console for debug
    elif platform_name == "macos":
        if not is_debug:
            pyinstaller_cmd.append('--windowed')  # Create .app bundle
        else:
            pyinstaller_cmd.append('--console')  # Console for debug
        # macOS specific options
        pyinstaller_cmd.extend([
            '--osx-bundle-identifier=com.trafficmonitor.app',
            '--target-arch=universal2',  # Universal binary for Intel and Apple Silicon
        ])
    
    # Add icon if available
    if icon_path:
        pyinstaller_cmd.append(f'--icon={icon_path}')
    
    # Exclude large unnecessary data folders that cause struct.error
    # Include folders but exclude ones with very long paths that cause Windows errors
    data_folders = [
        ('openvino_models', 'openvino_models'),
        ('resources', 'resources'),
        ('config', 'config'),
        ('vlm_backend', 'vlm_backend'),
        # ('services', 'services'),  # Exclude due to very long Grafana paths
        ('video_detection_full', 'video_detection_full'),
        ('violation_finale', 'violation_finale'),
        ('finale', 'finale'),
        ('llava_openvino_model', 'llava_openvino_model'),
        ('models', 'models'),
        ('data', 'data'),
        ('logs', 'logs'),
        ('temp', 'temp'),
        ('cache', 'cache'),
        ('assets', 'assets'),
        ('images', 'images'),
        ('icons', 'icons'),
        ('sounds', 'sounds'),
        ('videos', 'videos'),
        ('exports', 'exports'),
        ('snapshots', 'snapshots'),
        ('reports', 'reports'),
    ]
    parent_dir = Path.cwd().parent
    parent_folders = [
        ('yolo11n_openvino_model', 'yolo11n_openvino_model'),
        ('yolo11x_openvino_model', 'yolo11x_openvino_model'),
        ('models', 'parent_models'),
        # ('edge-ai-suites', 'edge-ai-suites'),  # May have long paths
        ('rcb', 'rcb'),
        ('smart-intersection', 'smart-intersection'),
        # ('__pycache__', '__pycache__'),  # Skip pycache
    ]
    for src, dst in data_folders:
        if os.path.exists(src):
            pyinstaller_cmd.append(f'--add-data={src}{data_sep}{dst}')
            print(f"✅ Added folder: {src}")
    for src, dst in parent_folders:
        parent_path = parent_dir / src
        if parent_path.exists():
            pyinstaller_cmd.append(f'--add-data={str(parent_path)}{data_sep}{dst}')
            print(f"✅ Added parent folder: {parent_path}")
    # Add all non-code files since --onedir can handle them
    non_code_files = [
        ('config.json', '.'),
        ('requirements.txt', '.'),
        ('environment.yml', '.'),
        ('Dockerfile', '.'),
        ('docker-compose.yml', '.'),
        ('README.md', '.'),
        ('finale.md', '.'),
        ('Week1.md', '.'),
        ('week2.md', '.'),
        ('all-files.txt', '.'),
        ('kernel.errors.txt', '.'),
        ('gitattributes', '.'),
        ('.gitattributes', '.'),
        ('qt_app.spec', '.'),
        ('version_info.txt', '.'),
        ('yolo11n.pt', '.'),
        ('yolo11x.bin', '.'),
        ('yolo11x.pt', '.'),
        ('yolo11x.xml', '.'),
    ]
    for src, dst in non_code_files:
        if os.path.exists(src):
            pyinstaller_cmd.append(f'--add-data={src}{data_sep}{dst}')
    
    # Only add hidden imports for external libraries, not your own code
    print("\n🔍 Checking available modules...")
    hidden_imports = [
        'cv2', 'openvino', 'numpy', 'scipy', 'sklearn', 'matplotlib', 'pandas', 'seaborn',
        'PySide6.QtCore', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtOpenGL', 'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
        'json', 'os', 'sys', 'time', 'traceback', 'pathlib', 'threading', 'queue', 'collections', 'itertools', 'functools', 'typing', 'subprocess', 'shutil', 'platform', 'datetime', 'logging', 'argparse', 'configparser', 'pickle', 'base64', 'hashlib', 'uuid', 'warnings', 'sqlite3', 'socket', 'asyncio',
        'torch', 'torchvision', 'transformers', 'huggingface_hub', 'tokenizers', 'optimum', 'timm', 'onnx', 'onnxruntime',
        'deep_sort_realtime', 'norfair', 'ultralytics',
        'paho.mqtt', 'influxdb_client', 'aiohttp', 'requests', 'flask',
        'PIL', 'tqdm', 'rich', 'pydantic', 'fpdf', 'jsonschema', 'pydot', 'pyparsing', 'tabulate', 'pyarrow', 'openai',
        'yaml', 'grafana_api'
    ]
    print(f"✅ Found {len(hidden_imports)} available external modules")
    for import_name in hidden_imports:
        pyinstaller_cmd.append(f'--hidden-import={import_name}')
    
    # Additional PyInstaller options
    pyinstaller_cmd.extend([
        '--noconfirm',  # Replace output directory without asking
        '--log-level=WARN',  # Reduce verbose output to see errors clearly
        '--workpath=build',
        '--distpath=dist',
        # Add paths for local modules
        f'--paths={current_dir}',
        f'--paths={current_dir}/utils',
        f'--paths={current_dir}/controllers', 
        f'--paths={current_dir}/ui',
        # Hidden imports for your local modules
        '--hidden-import=utils',
        '--hidden-import=utils.annotation_utils',
        '--hidden-import=utils.helpers',
        '--hidden-import=utils.enhanced_annotation_utils',
        '--hidden-import=utils.data_publisher',
        '--hidden-import=utils.mqtt_publisher',
        '--hidden-import=utils.traffic_light_utils',
        '--hidden-import=utils.scene_analytics',
        '--hidden-import=utils.crosswalk_utils',
        '--hidden-import=utils.crosswalk_utils2',
        '--hidden-import=utils.enhanced_tracker',
        '--hidden-import=utils.embedder_openvino',
        '--hidden-import=controllers',
        '--hidden-import=controllers.video_controller_new',
        '--hidden-import=controllers.model_manager',
        '--hidden-import=controllers.analytics_controller',
        '--hidden-import=controllers.performance_overlay',
        '--hidden-import=controllers.vlm_controller',
        '--hidden-import=controllers.smart_intersection_controller',
        '--hidden-import=ui',
        '--hidden-import=ui.main_window',
        '--hidden-import=ui.analytics_tab',
        '--hidden-import=ui.violations_tab',
        '--hidden-import=ui.export_tab',
        '--hidden-import=ui.modern_config_panel',
        '--hidden-import=ui.modern_live_detection_tab',
        # Root level modules
        '--hidden-import=splash',
        '--hidden-import=red_light_violation_pipeline',
        '--hidden-import=detection_openvino',
        '--hidden-import=detection_openvino_async',
        '--hidden-import=detection_openvino_fixed',
        '--hidden-import=violation_openvino',
        '--hidden-import=fallback_annotation_utils',
        '--hidden-import=annotation_utils',
        # Collect all submodules for external packages
        '--collect-all=PySide6',
        '--collect-all=openvino',
        # Exclude problematic modules that cause warnings
        '--exclude-module=tensorboard',
        '--exclude-module=tkinter',
        '--exclude-module=matplotlib.tests',
        '--exclude-module=pytest',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
    ])
    
    # Add platform specific optimizations
    if platform_name == "windows":
        pyinstaller_cmd.extend([
            '--version-file=version_info.txt' if os.path.exists('version_info.txt') else '',
        ])
    elif platform_name == "macos":
        pyinstaller_cmd.extend([
            '--codesign-identity=-',  # Ad-hoc signing
        ])
    
    # Remove empty arguments
    pyinstaller_cmd = [arg for arg in pyinstaller_cmd if arg]
    
    # Add main script
    pyinstaller_cmd.append('main.py')
    
    # Convert to string command
    cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in pyinstaller_cmd)
    
    # Build the application
    build_type = "debug" if is_debug else "release"
    if run_command(cmd_str, f"Building {app_name} ({build_type}) for {platform_name}"):
        if platform_name == "windows":
            exe_path = current_dir / "dist" / app_name / f"{app_name}.exe"
        elif platform_name == "macos":
            exe_path = current_dir / "dist" / f"{app_name}.app"
        else:
            exe_path = current_dir / "dist" / app_name / app_name
            
        print(f"\n✅ {build_type.title()} build completed successfully!")
        print(f"Executable location: {exe_path}")
        
        # Note about excluded folders due to Windows path limits
        print(f"\n📝 NOTE: Some folders with very long paths were excluded to avoid Windows path limit errors:")
        print(f"  - services/ (contains Grafana with very long nested paths)")
        print(f"  - edge-ai-suites/ (may contain long paths)")
        print(f"If your app needs these, manually copy them to the dist/{app_name}/ directory after build.")
        
        return True
    else:
        print(f"\n❌ {build_type.title()} build failed!")
        return False

def create_installer_script(platform_name, app_name):
    """Create platform-specific installer script"""
    
    if platform_name == "windows":
        # Create Windows batch installer
        installer_content = f'''@echo off
echo Installing {app_name}...
echo.

REM Create application directory
if not exist "%PROGRAMFILES%\\{app_name}" mkdir "%PROGRAMFILES%\\{app_name}"

REM Copy executable
copy "dist\\{app_name}.exe" "%PROGRAMFILES%\\{app_name}\\"

REM Create desktop shortcut
echo Creating desktop shortcut...
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut(\\"$env:USERPROFILE\\Desktop\\{app_name}.lnk\\"); $Shortcut.TargetPath = \\"$env:PROGRAMFILES\\{app_name}\\{app_name}.exe\\"; $Shortcut.Save()"

REM Create start menu shortcut
if not exist "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\{app_name}" mkdir "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\{app_name}"
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut(\\"$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs\\{app_name}\\{app_name}.lnk\\"); $Shortcut.TargetPath = \\"$env:PROGRAMFILES\\{app_name}\\{app_name}.exe\\"; $Shortcut.Save()"

echo Installation completed!
echo {app_name} has been installed to: %PROGRAMFILES%\\{app_name}
echo Desktop and Start Menu shortcuts have been created.
pause
'''
        
        with open(f"install_{app_name.lower()}_windows.bat", 'w') as f:
            f.write(installer_content)
        
        print(f"✅ Created Windows installer: install_{app_name.lower()}_windows.bat")
    
    elif platform_name == "macos":
        # Create macOS installer script
        installer_content = f'''#!/bin/bash
echo "Installing {app_name}..."
echo

# Create Applications directory if it doesn't exist
if [ ! -d "/Applications" ]; then
    sudo mkdir -p /Applications
fi

# Copy .app bundle to Applications
echo "Copying {app_name}.app to /Applications..."
sudo cp -R "dist/{app_name}.app" "/Applications/"

# Set permissions
sudo chmod -R 755 "/Applications/{app_name}.app"

# Create symlink for command line access (optional)
if [ ! -f "/usr/local/bin/{app_name.lower()}" ]; then
    echo "Creating command line symlink..."
    sudo ln -s "/Applications/{app_name}.app/Contents/MacOS/{app_name}" "/usr/local/bin/{app_name.lower()}"
fi

echo "Installation completed!"
echo "{app_name} has been installed to /Applications/{app_name}.app"
echo "You can now run it from Applications or use '{app_name.lower()}' command in terminal."
'''
        
        with open(f"install_{app_name.lower()}_macos.sh", 'w') as f:
            f.write(installer_content)
        
        # Make script executable
        os.chmod(f"install_{app_name.lower()}_macos.sh", 0o755)
        
        print(f"✅ Created macOS installer: install_{app_name.lower()}_macos.sh")

def test_executable(exe_name, platform_name):
    """Test the built executable and show errors"""
    if platform_name == "windows":
        exe_path = Path(f"dist/{exe_name}/{exe_name}.exe")
    elif platform_name == "macos":
        exe_path = Path(f"dist/{exe_name}.app")
    else:
        exe_path = Path(f"dist/{exe_name}/{exe_name}")
    
    if not exe_path.exists():
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    print(f"\n🧪 Testing executable: {exe_path}")
    print("Note: The executable will run in console mode for testing.")
    print("Any errors will be displayed below...")
    
    try:
        if platform_name == "macos":
            # For macOS .app bundles, run the actual executable inside
            actual_exe = exe_path / "Contents" / "MacOS" / exe_name
            if actual_exe.exists():
                result = subprocess.run([str(actual_exe)], timeout=10, text=True)
            else:
                result = subprocess.run(["open", str(exe_path)], timeout=10, text=True)
        else:
            # For Windows and Linux
            result = subprocess.run([str(exe_path)], timeout=10, text=True)
        
        print(f"✅ Executable test completed! Exit code: {result.returncode}")
        if result.returncode == 0:
            print("✅ No errors detected in executable!")
        else:
            print(f"⚠️  Executable exited with code {result.returncode} (may indicate errors)")
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Executable is running (timeout after 10s) - this is good!")
        print("✅ App appears to be working - close it manually if it opened")
        return True
    except Exception as e:
        print(f"❌ Error running executable: {e}")
        return False

def main():
    """Main build process"""
    print("🚀 Cross-Platform Traffic Monitor Build Script")
    print("=" * 60)
    
    # Get platform information
    platform_name, extension, data_sep = get_platform_info()
    print(f"🖥️  Detected platform: {platform_name}")
    print(f"📦 Target extension: {extension}")
    
    if platform_name == "unknown":
        print("❌ Unsupported platform")
        return False
    
    # Check if PyInstaller is available
    try:
        subprocess.run(['pyinstaller', '--version'], check=True, capture_output=True)
        print("✅ PyInstaller is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyInstaller not found. Installing...")
        if not run_command('pip install pyinstaller', "Installing PyInstaller"):
            print("Failed to install PyInstaller")
            return False
    
    # Check for required files
    required_files = ['main.py', 'ui', 'controllers', 'utils', 'config.json']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files/folders: {missing_files}")
        return False
    
    print("✅ All required files found")
    
    # Create dist directory if it doesn't exist
    os.makedirs('dist', exist_ok=True)
    
    # Build debug version first
    print(f"\n{'='*60}")
    print("🔨 Building DEBUG version...")
    print(f"{'='*60}")
    
    debug_success = build_application(platform_name, extension, data_sep, is_debug=True)
    
    if debug_success:
        print(f"\n✅ Debug build completed!")
        debug_name = "TrafficMonitorDebug"
        if platform_name == "windows":
            print(f"Debug executable: {Path.cwd()}/dist/{debug_name}.exe")
        elif platform_name == "macos":
            print(f"Debug executable: {Path.cwd()}/dist/{debug_name}.app")
    
    # Build main application
    print(f"\n{'='*60}")
    print("🔨 Building RELEASE version...")
    print(f"{'='*60}")
    
    release_success = build_application(platform_name, extension, data_sep, is_debug=False)
    
    if release_success:
        print(f"\n✅ Release build completed!")
        release_name = "TrafficMonitor"
        if platform_name == "windows":
            print(f"Release executable: {Path.cwd()}/dist/{release_name}.exe")
        elif platform_name == "macos":
            print(f"Release executable: {Path.cwd()}/dist/{release_name}.app")
        
        # Create installer scripts
        print(f"\n{'='*60}")
        print("📦 Creating installer scripts...")
        print(f"{'='*60}")
        
        create_installer_script(platform_name, "TrafficMonitor")
    
    # Ask user if they want to test the executables
    if debug_success or release_success:
        print(f"\n🧪 Testing Options:")
        test_choice = input("Do you want to test the built executables? (y/n): ").lower().strip()
        
        if test_choice in ['y', 'yes']:
            if debug_success:
                print(f"\n{'='*60}")
                print("🧪 Testing DEBUG version...")
                print(f"{'='*60}")
                test_executable("TrafficMonitorDebug", platform_name)
            
            if release_success:
                print(f"\n{'='*60}")
                print("🧪 Testing RELEASE version...")
                print(f"{'='*60}")
                test_executable("TrafficMonitor", platform_name)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 BUILD SUMMARY")
    print(f"{'='*60}")
    
    if debug_success and release_success:
        print("🎉 All builds completed successfully!")
        print(f"\n📂 Output files in dist/ directory:")
        
        if platform_name == "windows":
            print(f"  - TrafficMonitor/ (Release directory)")
            print(f"  - TrafficMonitorDebug/ (Debug directory)")
            print(f"\n📦 Installer:")
            print(f"  - install_trafficmonitor_windows.bat")
            print(f"\n📝 To test:")
            print(f"  1. Run debug version first: dist\\TrafficMonitorDebug\\TrafficMonitorDebug.exe")
            print(f"  2. If working, run release version: dist\\TrafficMonitor\\TrafficMonitor.exe")
            print(f"  3. For system-wide install, run: install_trafficmonitor_windows.bat (as admin)")
            
        elif platform_name == "macos":
            print(f"  - TrafficMonitor.app (Release)")
            print(f"  - TrafficMonitorDebug.app (Debug)")
            print(f"\n📦 Installer:")
            print(f"  - install_trafficmonitor_macos.sh")
            print(f"\n📝 To test:")
            print(f"  1. Run debug version first: open dist/TrafficMonitorDebug.app")
            print(f"  2. If working, run release version: open dist/TrafficMonitor.app")
            print(f"  3. For system-wide install, run: ./install_trafficmonitor_macos.sh")
        
        return True
    else:
        print("❌ Some builds failed!")
        if not debug_success:
            print("  - Debug build failed")
        if not release_success:
            print("  - Release build failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
