import sys
import platform
import importlib
from pathlib import Path

print("\n===== OpenVINO Environment Diagnostic =====\n")

# ------------------ SYSTEM INFO ------------------
print("System Information")
print("------------------")
print("OS:", platform.system(), platform.release())
print("Machine:", platform.machine())
print("Processor:", platform.processor())

# ------------------ PYTHON INFO ------------------
print("\nPython Information")
print("------------------")
print("Python version:", sys.version)

# ------------------ OPENVINO INFO ------------------
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

# ------------------ PACKAGE CHECK ------------------
print("\nPackage Check")
print("------------------")

req_file = Path("requirements.txt")

# mapping pip names → import names
IMPORT_MAP = {
    "pillow": "PIL",
    "opencv-python": "cv2",
    "optimum-intel": "optimum",
}

def parse_line(line):
    """Extract package name + required version"""
    if "==" in line:
        name, version = line.split("==")
    elif ">=" in line:
        name, version = line.split(">=")
    else:
        name, version = line, None

    name = name.strip()
    version = version.strip() if version else None

    # remove extras like qrcode[pil]
    name = name.split("[")[0]

    return name, version

if req_file.exists():
    packages = []

    with open(req_file) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("--"):
                continue

            pkg_name, req_version = parse_line(line)
            packages.append((pkg_name, req_version))

    for pkg, req_version in packages:
        module_name = IMPORT_MAP.get(pkg, pkg.replace("-", "_"))

        try:
            module = importlib.import_module(module_name)

            # try to get installed version
            installed_version = getattr(module, "__version__", None)

            if req_version and installed_version:
                if installed_version.startswith(req_version):
                    print(f"{pkg} ✔ installed ({installed_version})")
                else:
                    print(f"{pkg} ⚠ version mismatch (installed: {installed_version}, required: {req_version})")
            else:
                print(f"{pkg} ✔ installed")

        except ImportError:
            print(f"{pkg} ❌ missing")

else:
    print("No requirements.txt found.")

print("\n===========================================\n")