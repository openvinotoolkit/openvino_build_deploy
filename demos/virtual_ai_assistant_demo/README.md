# Virtual AI Assistant with OpenVINO™

The demo works as a virtual AI assistant. The default personality is a healthcare assistant, whose task is to greet the patient and summarize the patient's condition based on the conversation and uploaded examination report.

The personality and behaviour can be easily changed with the config file (see Step 2).

![](https://github.com/user-attachments/assets/d3ede3e5-d6af-46c7-8e00-250fc2137ed3)

Here are the steps involved in this demo:

Step 0: Install Python and prerequisites

Step 1: Set up the environment

Step 2: Create a YAML Personality File

Step 3: Run the Application

Now, let's dive into the steps starting with installing Python.

## Step 0: Install Python and prerequisites 

This project requires Python 3.8 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git git-lfs gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you will probably need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Step 1: Set up the environment

1. Clone the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/demos/virtual_ai_assistant_demo
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

## Step 2: Create a YAML Personality File

You can create a personality file for your virtual AI assistant using YAML. Each personality can be customized based on the specific role of the assistant, such as a health assistant, bartender, or legal assistant. 

### Key Components of a Personality File

A typical YAML personality file has the following sections:

1. **Instructions**: A brief, descriptive title for the assistant.
2. **System Configuration**: Instructions that define the assistant's behavior and limitations.
3. **Greet the User Prompt**: The first interaction where the assistant introduces itself.
4. **Summarize the User Prompt**: Instructions on how the assistant summarizes the conversation.

### Some tips for creating this YAML file: 

The instructions provides an introduction to the assistant, along with the title and important notes for the user. It should be clear and concise, giving users context on how to interact with the assistant.

```yaml
instructions: | 
  # [Assistant Name]: [Brief Role Description]

        Instructions for use:  
        1. Provide a brief step-by-step guide on how the assistant works.  
        2. Include key points the user should know before interacting with the assistant.  
        3. Mention any important disclaimers, if applicable.

        **Note: [Add a disclaimer or key note about what the assistant can and cannot do].**
```


## Step 3: Run the Application

To run the application, use the following command. The application will start downloading Llama 3, and Distil-Whisper by default. 
To use LLama3, you must provide your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens) as a parameter. 
It will take up to an hour (depending on your internet speed) for the first time running this application due to the large downloads and conversion of the models. 
Once the models are cached, the subsequent executions will be much faster.

```shell
python main.py --hf_token [replace-with-your-token]
```

Change the personality and behaviour by providing a new YAML config file:

```shell
python main.py --personality use_your_personality.yaml
```

You can also change chat, asr or embedding model:

```shell
python main.py --asr_model distil-whisper/distil-large-v2 --chat_model OpenVINO/Phi-3-medium-4k-instruct-int4-ov --embedding_model BAAI/bge-large-en-v1.5
```

Running with `--public` will allow you to access from any computer:

```shell
python main.py --public
```
