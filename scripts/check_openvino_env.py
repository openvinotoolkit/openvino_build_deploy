import sys
import importlib

print("\n===== OpenVINO Environment Diagnostic =====\n")

# Python version
print("Python version:", sys.version)

# Check OpenVINO installation
try:
    import openvino as ov
    core = ov.Core()
    print("\nOpenVINO version:", ov.__version__)
    print("Available devices:")
    for device in core.available_devices:
        print(" -", device)
except Exception as e:
    print("\nOpenVINO check failed:", e)

# Check required packages
packages = ["numpy", "onnx", "gradio", "opencv-python", "torch", "transformers"]

print("\nChecking required packages:\n")

for pkg in packages:
    if importlib.util.find_spec(pkg):
        print(pkg, "✔ installed")
    else:
        print(pkg, "❌ missing")

print("\n===========================================\n")