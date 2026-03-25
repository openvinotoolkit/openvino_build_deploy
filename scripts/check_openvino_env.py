import sys
import platform
import importlib

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

# OpenVINO
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

# Required packages
packages = ["numpy", "onnx", "gradio", "opencv-python", "torch", "transformers"]

print("\nPackage Check")
print("------------------")

for pkg in packages:
    if importlib.util.find_spec(pkg):
        print(pkg, "✔ installed")
    else:
        print(pkg, "❌ missing")

print("\n===========================================\n")