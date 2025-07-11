# Running YOLOv8 Object Detection with ONNX and OpenVINO

Warning: This demo requires **Python 3.10-3.13**.

In this demo, we'll perform object detection leveraging YOLOv8 with Ultralytics, and with ONNX using the OpenVINO Execution Provider for enhanced performance, to detect up to 80 different objects (e.g., birds, dogs, etc.)
This sample was modified from one of the [available Onnx Runtime Inference examples here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/python/OpenVINO_EP/yolov8_object_detection). 

<p align="center">
    <img src="https://github.com/user-attachments/assets/a3e35991-0c3b-47e0-a94a-c70d7b135261"/>
</p>


### Installation Instructions
- Create a virtual environment using 
  ```sh  
  python -m venv <venv-name>
  ```
- To activate the virtual environment use
  ```sh
  \<venv-name>\Scripts\activate
  ```
- Install the required dependencies via pip
  ```sh
  pip install -r requirements.txt
  ```
- Now you only need a Jupyter server to start.
  ```sh
  jupyter lab
  ```

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=notebooks/onnxruntime_yolov8&file=README.md" />