import sys
import platform
import importlib
from pathlib import Path

print("\n===== OpenVINO Environment Diagnostic =====\n")

# System info
print("System Information")
print("------------------")
print("OS:", platform.system(), platform.release())
print("Machine:", platform.machine())
print("Processor:", platform.processor())

# Python info
print("\nPython Information")
print("------------------")
print("Python version:", sys.version)

# OpenVINO info
print("\nOpenVINO Information")
print("------------------")

try:
    import openvino as ov
    core = ov.Core()
    print("OpenVINO version:", ov.__version__)
    print("Available devices:")
    for device in core.available_devices:
        print(" -", device)
except Exception as e:
    print("OpenVINO check failed:", e)

# Check packages from requirements.txt
print("\nPackage Check")
print("------------------")

req_file = Path("requirements.txt")

if req_file.exists():
    packages = []

    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pkg = line.split("==")[0].split(">=")[0].split("<=")[0]
                packages.append(pkg)

    for pkg in packages:
        try:
            importlib.import_module(pkg)
            print(pkg, "✔ installed")
        except ImportError:
            print(pkg, "❌ missing")
else:
    print("No requirements.txt found. Skipping package checks.")

print("\n===========================================\n")