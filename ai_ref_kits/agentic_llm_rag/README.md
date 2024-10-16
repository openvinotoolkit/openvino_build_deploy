<div id="top" align="center">
  <h1>Agentic LLMs and RAG with the OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video</a> - Placeholder
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

The Agentic AI RAG Chatbot utilizes the OpenVINO™ toolkit to create a streamlined, voice-activated interface that consumers and retail employees can use as a smart, personalized retail assistant capable of performing math, understanding instructional documents, and holding conversations. At its core, the application harnesses models for speech recognition and text-to-speech (TTS) synthesis. It is configured to understand user prompts, engage in meaningful dialogue via Large Language Model-powered agents, understand guides and information uploaded to it via Retrieval-Augmented Generation (RGA), and provide spoken responses, making it an interactive and user-friendly personalized assistant that simulates the experience of a interactive kiosk.

This kit uses the following technology stack:
- [OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([Docs](https://docs.openvino.ai/))
- [Meta’s Llama](https://llama.meta.com/llama3/)

Check out our [Edge AI Reference Kits repository](/) for other kits.

![agentic-ai-llm-rag](tbd)

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting up your Environment](#setting-up-your-environment)
  - [How to Access LlaMA](#how-to-access-llama)
  - [Running the Application](#running-the-application)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Installing Prerequisites

This project requires Python 3.8 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you may also need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)._

## Setting up your Environment

### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/ai_ref_kits/agentic_llm_rag
```

### Creating a Virtual Environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m venv venv
```
This will create a new virtual environment named "venv" in the current directory.

### Activating the Environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating systems such as Linux or macOS
```

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```
## How to Access LlaMA

_NOTE: If you already have access to the LlaMA model weights, skip to the authentication step, which is mandatory for converting the LlaMA model._

## Model Conversion and Optimization

_NOTE: This reference kit requires much bandwidth and disk space (>8GB) for downloading models. Also, the conversion may take much time (>2h) and need much memory (>32GB)) when running for the first time as the models used here are huge. After the first run, the subsequent runs will be done much faster._

The application uses three separate models for its operation, each requiring conversion and optimization for use with OpenVINO™. Follow the order below to convert and optimize each model:
1. **Automated Speech Recognition Distil-Whisper Conversion**:
The ASR model is responsible for converting spoken language (audio) into written text. This functionality is crucial as it enables the chatbot to understand and process voice-based user queries.
```shell
python convert_and_optimize_asr.py --asr_model_type distil-whisper-large-v3 --precision int8
```
This script will convert and optimize the automatic speech recognition (ASR) model performing weights quantization.

2. **Chat , Embedding, and Reranker Model Conversion**:
The chat model is at the core of the chatbot's ability to generate meaningful and context-aware responses. It processes the text input from the ASR model and produces a human-like response.
The embedding model represents text data (both user queries and potential responses or knowledge base entries) as numerical vectors. These vectors are essential for tasks such as semantic search and similarity matching.
The reranker model is used in retrieval-augmented generation (RAG) setups to reorder or "rerank" retrieved results, ensuring the most relevant information is presented to the user.
To convert the chat, embedding, and reranker models, run the following command:
```shell
python convert_and_optimize_chat.py --chat_model_type llama3.1-8B --embedding_model_type bge-large --reranker_model_type bge-reranker-large --precision int4 --hf_token your_huggingface_token --model_dir model
```
This script will handle the conversion and optimization of:

- The chat model (`llama3.1-8B`) with `int4` precision.
- The embedding model (`bge-large`) with `FP32` precision.
- Reranker Model (`bge-reranker-large`) with `FP32` precision.

    The script requires a Hugging Face token (`--hf_token`) for authentication, which allows access to gated models like LLaMA. The converted models will be saved in the specified `model` directory.

    To access the original LlaMA model weights:
    Accept the License on Hugging Face: Visit the LlaMA model page, for example [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) on Hugging Face. Read and accept the license. Once you have accepted the license, you'll gain access to the LlaMA models. Note that requests used to take up to one hour to get processed.

After running the conversion scripts, you can run app.py to launch the application.

## Running the Application (Gradio Interface)

_NOTE: This application requires much memory (>16GB) as the models used here are huge (especially chatbot). It may also be slow for less powerful devices._

Execute the `app.py` script with the following command, including all necessary model directory arguments:
```shell
python app.py \
  --personality concierge_personality.yaml \
  --asr_model path/to/asr_model \
  --chat_model path/to/chat_model \
  --embedding_model path/to/embedding_model \
  --vocoder_model vocoder_model_name \
  --public
```
- `--personality path/to/personality.yaml`: Path to your custom personality YAML file (e.g., `concierge_personality.yaml`). This file defines the assistant's personality, including instructions, system configuration, and greeting prompts. Feel free to create and provide your own custom personality file.

- `--asr_model path/to/asr_model`: Path to your ASR (Automatic Speech Recognition) model directory, using `int8` precision (e.g., `model/distil-whisper-large-v3-int8`) for efficient speech recognition.

- `--chat_model path/to/chat_model`: Path to your chat model directory (e.g., `model/llama3.1-8B-INT4`) that drives conversation flow and response generation.

- `--embedding_model path/to/embedding_model`: Path to your embedding model directory (e.g., `model/bge-small-FP32`) for understanding and matching text inputs.

- `--vocoder_model vocoder_model_name`: HuggingFace name of your vocoder model (e.g., `microsoft/speecht5_hifigan`). Enhances audio quality of the spoken responses.

- `--public`: Include this flag to make the Gradio interface publicly accessible over the network. Without this flag, the interface will only be available on your local machine.

### Create a Custom YAML Personality File

You can create a personality file for your virtual AI assistant using YAML. Each personality can be customized based on the specific role of the assistant, such as a concierge, bartender, or legal assistant. 

#### Key Components of a Personality File

A typical YAML personality file has the following sections:

1. **Instructions**: A brief, descriptive title for the assistant.
2. **System Configuration**: Instructions that define the assistant's behavior and limitations.
3. **Greet the User Prompt**: The first interaction where the assistant introduces itself.

#### Some tips for creating this YAML file: 

The instructions provide an introduction to the assistant, along with the title and important notes for the user. It should be clear and concise, giving users context on how to interact with the assistant.

```yaml
instructions: | 
  # [Assistant Name]: [Brief Role Description]

        Instructions for use:  
        1. Provide a brief step-by-step guide on how the assistant works.  
        2. Include key points the user should know before interacting with the assistant.  
        3. Mention any important disclaimers, if applicable.

        **Note: [Add a disclaimer or key note about what the assistant can and cannot do].**
```

### Accessing the Web Interface
After running the script, Gradio will provide a local URL, typically `http://127.0.0.1:XXXX`, which you can open in your web browser to start interacting with the assistant. If you configured the application to be accessible publicly, Gradio will also provide a public URL.

Trying Out the Application
 TBD

Feel free to engage with the Chatbot, ask questions, or give commands as per the assistant's capabilities. This hands-on experience will help you understand the assistant's interactive quality and performance.

Enjoy exploring the capabilities of your Chatbot!

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>
