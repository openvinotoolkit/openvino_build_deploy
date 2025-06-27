# Narrator Demo with OpenVINO™

This demo generates natural language captions for images or video streams using the BLIP (Bootstrapped Language Image Pretraining) model, accelerated by OpenVINO™. The application can process live webcam feeds or video files, displaying the generated captions in real time. You can switch between different devices (CPU, GPU, etc.) on the fly to compare performance.

![narrator_demo](https://github.com/user-attachments/assets/e2a3ed34-93ff-4aaa-87cd-e6ad61eaf421)

## Quick Launch using Setup Scripts

If you want a **quick setup** without manually installing dependencies, use the provided installer scripts in the `setup` directory. These scripts will **automatically configure** everything needed to run the Narrator Demo.

### **For Windows**
1. Download or copy the `install.bat` and `run.bat` files to your local directory (or use them directly from the repo).
2. Double-click `install.bat` to install dependencies and set up the environment.
3. After installation, double-click `run.bat` to start the demo.

### **For Linux**
1. Download or copy the `install.sh` and `run.sh` files to your local directory (or use them directly from the repo).
2. First, ensure the installer scripts have execute permissions:
```shell
chmod +x install.sh run.sh
```
3. Run the installer to set up everything:
```shell
./install.sh
```
4. After installation, start the demo by running:
```shell
./run.sh
```
These scripts will handle cloning the repository, creating the virtual environment, and installing dependencies automatically. If you prefer a manual setup, follow Steps 1-5 below.

## Manual Environment Setup

### Step 0

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

### Step 1: Install Python and Prerequisites

This project requires Python 3.10-3.13 and a few libraries. If you don't have Python installed, download it from https://www.python.org/downloads/ and follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install system libraries and tools:

```shell
sudo apt install git python3-venv python3-dev
```

_NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

### Step 2: Clone the Repository

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
cd openvino_build_deploy/demos/narrator_demo
```

### Step 3: Create and Activate a Virtual Environment

```shell
python3 -m venv venv
source venv/bin/activate   # For Unix-based operating systems
```
_On Windows, use:_
```shell
venv\Scripts\activate
```

### Step 4: Install Python Packages

```shell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 5: Run the Application

To run the application on your webcam (default):

```shell
python main.py --stream 0
```

To run it on a specific video file:

```shell
python main.py --stream input.mp4
```

You can select which BLIP model to use (base or large):

```shell
python main.py --stream 0 --model_name Salesforce/blip-image-captioning-large
```

The first run will automatically download and convert the required BLIP model to OpenVINO IR format. The converted models will be stored in the `model` directory for future runs.

To see all available options:

```shell
python main.py --help
```

## Controls

- **ESC** or **q**: Exit the application.
- **1, 2, ...**: Switch between available devices (CPU, GPU, etc.) if supported.

## Output

- The application displays the video stream with generated captions overlaid.
- Inference time and current device/model are shown on the video.

## Notes

- The demo uses the BLIP model from HuggingFace Transformers and requires downloading model weights on first run.
- The `model` directory will contain the OpenVINO IR models after conversion.
- For best performance, use a machine with an Intel CPU, GPU or NPU.

---

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=demos/narrator_demo&file=README.md" />

---
