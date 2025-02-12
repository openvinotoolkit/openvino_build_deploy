# Spot a Defect with OpenVINO™

The demo detects, tracks and counts defined objects in front of the webcam. The default object is a hazelnut, but it can be changed to any other object. It works especially good with a conveyor belt.

![image](https://github.com/user-attachments/assets/103617f8-e895-4cc0-9ed9-60a1e87b8706)

Here are the steps involved in this demo:

Step 0: Install Python and prerequisites

Step 1: Set up the environment

Step 2: Run the Application

Now, let's dive into the steps starting with installing Python.

## Step 0

This project requires Python 3.10-3.12 and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git python3-venv python3-dev
```

_NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Step 1

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/demos/spot-a-defect-demo
```

2. Create a virtual environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m venv venv
```
This will create a new virtual environment named "venv" in the current directory.

3. Activate the environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating system such as Linux or macOS
```

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

4. Install the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

## Step 2

To run the application, use the following command:

```shell
python main.py --stream 0
```
And you can run it on specific video input

```shell
python main.py --stream input.mp4
```

By default, the YOLOv8s-Worldv2 model is used. To change this, select another model from the family:

```shell
python main.py --stream 0 --model_name yolov8x-world
```

To change the inference device use the `--device` option. By default, AUTO is used.

```shell
python main.py --stream 0 --device GPU
```

Run the following to see all available options.

```shell
python main.py --help
```
