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

# Package check
print("\nPackage Check")
print("------------------")

req_file = Path("requirements.txt")

if req_file.exists():
    with open(req_file) as f:
        packages = [
            line.strip().split("==")[0].split(">=")[0].split("<=")[0]
            for line in f if line.strip() and not line.startswith("#")
        ]

    for pkg in packages:
        module_name = pkg.replace("-", "_")  # fix import naming
        try:
            importlib.import_module(module_name)
            print(pkg, "✔ installed")
        except ImportError:
            print(pkg, "❌ missing")
else:
    print("No requirements.txt found.")

print("\n===========================================\n")