"""
Simple launcher for the Traffic Monitoring application with enhanced controller.
Uses subprocess to avoid encoding issues.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the application using subprocess to avoid encoding issues."""
    print("\n" + "="*80)
    print("🚀 Launching Traffic Monitoring with Enhanced Controller")
    print("="*80)
    
    # Add parent directory to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    # Path to main.py
    main_script = Path(__file__).parent / "main.py"
    
    if not main_script.exists():
        print(f"❌ Error: {main_script} not found!")
        return 1
        
    print(f"✅ Launching {main_script}")
    
    # Launch the application using subprocess
    try:
        subprocess.run([sys.executable, str(main_script)], check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        return e.returncode
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
