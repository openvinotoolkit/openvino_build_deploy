# OpenVINO Demos

## How to Setup

1. Install the latest drivers here
- NPU Driver: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
- GPU Driver: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html

2. Install Python 3.11.x, or 3.12.x
- https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe

3. Enable PowerShell for venv usage
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```

### Install the packages (may take a while)
```
python -m venv openvino_venv
openvino_venv\Scripts\activate
python -m pip install --upgrade pip

pip install nncf==2.14.1 onnx==1.17.0 optimum-intel==1.22.0
pip install openvino==2025.1 openvino-tokenizers==2025.1 openvino-genai==2025.1

#for whisper demo
pip install pyaudio librosa
```
Note: Only OpenVINO 2025.1 is validated for this demo.

To validate the installation, run the following command and you should be able to see `[CPU, GPU, NPU]` in the list of available devices
```
python -c "from openvino import Core; print(Core().available_devices)"
```

Now your environment is ready for trying out the demos. 
- [chat sample](https://github.com/raymondlo84Fork/MSBuild2025/tree/main/openvino_genai/chat_sample)
- [whisper](https://github.com/raymondlo84Fork/MSBuild2025/tree/main/openvino_genai/whisper)

## References:
NPU with OpenVINO GenAI: https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
