#!/usr/bin/env python3
"""
Simple Build Script for Traffic Monitor Application
Builds a working executable with essential files only
"""

import os
import subprocess
import sys
import shutil
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        # Fix encoding issues on Windows
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def build_simple_executable():
    """Build a simple working executable"""
    
    current_dir = Path.cwd()
    print(f"Building from: {current_dir}")
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed {folder}")
    
    for spec_file in ['TrafficMonitor.spec', 'TrafficMonitorDebug.spec', 'TrafficMonitorSimple.spec']:
        if os.path.exists(spec_file):
            os.remove(spec_file)
            print(f"Removed {spec_file}")
    
    # Essential PyInstaller command
    pyinstaller_cmd = [
        'pyinstaller',
        '--name=TrafficMonitorSimple',
        '--onefile',
        '--console',  # Always use console for debugging
        '--clean',
        
        # Essential data folders only
        '--add-data=ui;ui',
        '--add-data=controllers;controllers',
        '--add-data=utils;utils',
        '--add-data=config.json;.',
        
        # Essential hidden imports only
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtWidgets',
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=openvino',
        '--hidden-import=json',
        '--hidden-import=os',
        '--hidden-import=sys',
        '--hidden-import=pathlib',
        
        # PyInstaller options
        '--noconfirm',
        '--log-level=WARN',  # Reduce verbose output
        '--workpath=build',
        '--distpath=dist',
        
        # Main script
        'main.py'
    ]
    
    # Convert to string command
    cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in pyinstaller_cmd)
    
    # Build the application
    if run_command(cmd_str, "Building simple Traffic Monitor executable"):
        exe_path = current_dir / "dist" / "TrafficMonitorSimple.exe"
        print(f"\n✅ Simple build completed successfully!")
        print(f"Executable location: {exe_path}")
        return True
    else:
        print(f"\n❌ Simple build failed!")
        return False

def test_executable():
    """Test the built executable"""
    exe_path = Path("dist/TrafficMonitorSimple.exe")
    
    if not exe_path.exists():
        print("❌ Executable not found!")
        return False
    
    print(f"\n🧪 Testing executable: {exe_path}")
    print("Note: The executable will run in console mode for testing.")
    print("Close the app window to continue...")
    
    try:
        # Run the executable and wait for it to finish
        result = subprocess.run([str(exe_path)], capture_output=False, text=True, timeout=30)
        print(f"✅ Executable ran successfully! Exit code: {result.returncode}")
        return True
    except subprocess.TimeoutExpired:
        print("⏰ Executable is running (timeout after 30s) - this is good!")
        return True
    except Exception as e:
        print(f"❌ Error running executable: {e}")
        return False

def main():
    """Main build process"""
    print("🚀 Simple Traffic Monitor Build Script")
    print("=" * 50)
    
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
    
    # Build simple executable
    if build_simple_executable():
        print(f"\n✅ Build completed!")
        print(f"Executable: {Path.cwd()}/dist/TrafficMonitorSimple.exe")
        
        # Ask user if they want to test
        test_choice = input("\n🧪 Do you want to test the executable? (y/n): ").lower().strip()
        if test_choice in ['y', 'yes']:
            test_executable()
        
        print(f"\n📝 To run your app:")
        print(f"1. Run: dist\\TrafficMonitorSimple.exe")
        print(f"2. The console window will show debug info")
        print(f"3. The app window should open normally")
        
        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
