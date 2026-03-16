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

# special cases where pip name != import name
IMPORT_NAME_MAP = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "optimum-intel": "optimum",
}

if req_file.exists():
    packages = []

    with open(req_file) as f:
        for line in f:
            line = line.strip()

            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # skip pip options
            if line.startswith("--"):
                continue

            # remove version constraints
            pkg = line.split("==")[0].split(">=")[0].split("<=")[0]

            # remove extras like qrcode[pil]
            pkg = pkg.split("[")[0]

            packages.append(pkg)

    for pkg in packages:

        # determine correct import name
        module_name = IMPORT_NAME_MAP.get(pkg, pkg.replace("-", "_"))

        try:
            importlib.import_module(module_name)
            print(pkg, "✔ installed")
        except ImportError:
            print(pkg, "❌ missing")

else:
    print("No requirements.txt found.")

print("\n===========================================\n")