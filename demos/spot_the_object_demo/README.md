# Spot the Object with OpenVINO™

The demo detects, tracks and counts defined objects in front of the webcam. The default object is a hazelnut, but it can be changed to any other object. It works especially good with a conveyor belt.

![spot_the_object](https://github.com/user-attachments/assets/e0b1f56a-a7b3-4bf0-a056-1fac804c2de3)

## Quick Launch using Setup Scripts

If you want a **quick setup** without manually installing dependencies, use the provided installer scripts. These scripts will **automatically configure** everything needed to run the Spot the Object Demo.

### **For Windows**
1. Download the `install.bat` and `run.bat` files to your local directory.
2. Double-click `install.bat` to install dependencies and set up the environment.
3. After installation, double-click `run.bat` to start the demo.

### **For Linux and MacOS**
1. Download the `install.sh` and `run.sh` files to your local directory.
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
These scripts will handle cloning the repository, creating the virtual environment, and installing dependencies automatically. If you prefer a manual setup, follow Steps 1-3 below.

## Manual Environment Setup

Here are the steps involved in this demo:

Step 1: Install Python and prerequisites

Step 2: Set up the environment

Step 3: Run the Application

Now, let's dive into the steps starting with installing Python.

## Step 0

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Step 1

This project requires Python 3.10-3.13 and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git python3-venv python3-dev
```

_NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Step 2

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/demos/spot_the_object_demo
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

## Step 3

To run the application, use the following command:

```shell
python main.py --stream 0
```

And you can run it on specific video input

```shell
python main.py --stream input.mp4
```

To change the class to detect, use the `--class_name` option. By default, hazelnut is used. You should also provide auxiliary classes to improve the detection.

```shell
python main.py --stream 0 --class_name hazelnut --aux_classes nut "brown ball"
```

```shell

By default, the YOLOE-11M model is used. To change this, select another model from the family:

```shell
python main.py --stream 0 --model_name yoloe-11s-seg
```

To change the inference device use the `--device` option. By default, AUTO is used.

```shell
python main.py --stream 0 --device GPU
```

Run the following to see all available options.

```shell
python main.py --help
```
[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=demos/spot_the_object_demo&file=README.md" />
