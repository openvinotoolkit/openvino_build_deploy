# Paint Your Dreams with OpenVINO™

The demo generates images in seconds on Intel hardware. There are many options to customize the demo behaviour:
- inference device
- number of steps
- image size
- guidance scale
- seed and its randomization

Check "Generate endlessly" to generate a new image just after the previous has appeared (a very attention-bringing thing).

![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/a7f53cf2-a40b-4eb2-bb9a-72969ce8ad04)

Here are the steps involved in this demo:

Step 0: Install Python and prerequisites

Step 1: Set up the environment

Step 2: Run the Application

Now, let's dive into the steps starting with installing Python.

**Input text:** a beautiful pink unicorn, 8k

![unicorn](https://user-images.githubusercontent.com/29454499/277367065-13a8f622-8ea7-4d12-b3f8-241d4499305e.png)

## Step 0

This project requires Python 3.10 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

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
cd openvino_build_deploy/demos/paint_your_dreams_demo
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
python main.py
```

During the first run (i.e., before generating the first image), the application will download the model online. Please wait until the process is completed and do not disconnect the internet or close the application. Additionally, the first run may take significantly longer due to caching and other behavior, and you will notice a significant speed-up after warm-up. 

To change the model precision to FP16, run:

```shell
python main.py --model_name OpenVINO/LCM_Dreamshaper_v7-fp16-ov
```

This change will provide speed up on CPU, and a slight improvement on GPU. Also, there is slight degradation in the image generation quality. You can increase the number of step by 1 or 2 to improve that.  

The demo will be available for localhost only (i.e., the application is not accessible to outside network). To make it available in the local network use:

```shell
python main.py --local_network
```

Run the following to see all available options.

```shell
python main.py --help
```
