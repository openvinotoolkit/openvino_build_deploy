# Theme demo with OpenVINO™

This is a funny demo making people look like something/someone else based on the chosen theme. It brings much attention to the booth. The available themes are: Christmas and Halloween.

With Christmas theme, the demo detects people in front of the webcam and changes them into Santa Claus (the biggest face) and reindeer (all other faces). The name of the reindeer depends on the face expression (emotion).

![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/0cbf768c-0260-41bc-af64-00dfdaa9110c)

When using the Halloween theme, the demo detects people in front of the webcam and changes them into skeletons with a pumpkin head.

![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/b289b9f0-1c5b-4cae-ae0b-ea905d05d5e5)

Here are the steps involved in this demo:

Step 1: Install Python and prerequisites

Step 2: Set up the environment

Step 3: Run the Application

Now, let's dive into the steps starting with installing Python. 

## Step 0

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Step 1

This project requires Python 3.10-3.12 and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

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
cd openvino_build_deploy/demos/theme_demo
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

To change the theme, use:

```shell
python main.py --stream 0 --theme halloween
```

Run the following to see all available options.

```shell
python main.py --help
```
