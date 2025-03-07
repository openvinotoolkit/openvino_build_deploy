
# Emotion-Based Music Recommendation System with OpenVINO™

This project utilizes OpenVINO for real-time emotion recognition from video streams. Based on the detected emotions, it plays music to match or improve the user's mood. The system runs as a background script, providing real-time feedback without interfering with day-to-day tasks.
## Features

- Real-time emotion detection
- Music recommendation based on detected emotions
- Motivational text display based on emotions
- Logging of detected emotions and dominant emotions

![happy](assets/happy.png)
![neutral](assets/neutral.png)
![sad Turn](assets/sad.png)

Here are the steps involved in this demo:

Step 1: Install Python and prerequisites

Step 2: Set up the environment

Step 3: Run the Application

Step 4: Add songs according to your mood in the respective emotions directories

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
cd openvino_build_deploy/demos/mood_based_music_recommendation_demo
```

2. Create a virtual environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python -m venv venv
```
This will create a new virtual environment named "venv" in the current directory.

3. Activate the environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating system such as Linux or macOS
```

_NOTE: If you are using Windows (Command Prompt), use `venv\Scripts\activate` command instead and use `venv\Scripts\Activate.ps1` for Windows (PowerShell).

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
python main.py --stream 0 --music_dir /path/to/music_directory
```

To change the models, precision or device use:

```shell
python main.py --stream 0 --device AUTO --emotion_model_name emotions-recognition-retail-0003 --model_precision FP32 --music_dir /path/to/music_directory
```

Run the following to see all available options.

```shell
python main.py --help
```