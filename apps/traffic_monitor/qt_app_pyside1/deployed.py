#!/usr/bin/env python3
"""
Simple Deployment Script for Traffic Monitoring System
====================================================

This script simply replaces main.py with a better version that loads main_window1.py first.

Bhai, no advanced features - just simple main.py edit!
"""

import os
import sys

def deploy_main_py():
    """Deploy simple enhanced version to main.py"""
    main_py_path = os.path.join(os.path.dirname(__file__), "main.py")
    backup_path = os.path.join(os.path.dirname(__file__), "main_backup.py")
    
    try:
        # Create backup of original main.py
        if os.path.exists(main_py_path):
            import shutil
            shutil.copy2(main_py_path, backup_path)
            print(f"✅ Backup created: {backup_path}")
        
        # Write the simple enhanced version to main.py
        enhanced_main_content = '''from PySide6.QtWidgets import QApplication
import sys
import os
import time

def main():
    # Create application instance first
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Show splash screen if available
    splash = None
    try:
        from splash import show_splash
        splash, app = show_splash(app)
    except Exception as e:
        print(f"Could not show splash screen: {e}")

    # Add a short delay to show the splash screen
    if splash:
        time.sleep(1)

    # Try to load UI with fallback - Modern UI first!
    try:
        # Try modern UI first (main_window1.py)
        print("🔄 Attempting to load MainWindow1 (Modern UI)...")
        from ui.main_window1 import MainWindow
        print("✅ SUCCESS: Using enhanced MainWindow1 with modern UI")
    except Exception as e:
        # Fall back to standard version
        print(f"⚠️ Could not load MainWindow1: {e}")
        print("🔄 Attempting fallback to standard MainWindow...")
        try:
            from ui.main_window import MainWindow
            print("✅ Using standard MainWindow")
        except Exception as e:
            print(f"❌ Could not load any MainWindow: {e}")
            sys.exit(1)

    try:
        # Initialize main window
        window = MainWindow()
        
        # Close splash if it exists
        if splash:
            splash.finish(window)
        
        # Show main window
        window.show()
        
        # Start application event loop
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open(main_py_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_main_content)
        
        print(f"✅ Enhanced main.py deployed successfully!")
        print(f"📝 Original main.py backed up to: {backup_path}")
        print(f"🎯 You can now run: python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to deploy main.py: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple Traffic Monitoring System Deployment")
    print("=" * 50)
    print()
    print("This will replace main.py to load main_window1.py first (Modern UI)")
    print()
    
    choice = input("Deploy enhanced main.py? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        print("\n📦 Deploying enhanced version to main.py...")
        if deploy_main_py():
            print("✅ Deployment successful!")
            print("🎯 Now run: python main.py")
        else:
            print("❌ Deployment failed!")
            sys.exit(1)
    else:
        print("\n👋 Goodbye!")
        sys.exit(0)
