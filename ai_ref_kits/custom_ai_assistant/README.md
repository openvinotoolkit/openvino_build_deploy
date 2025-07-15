<div id="top" align="center">
  <h1>Custom AI Assistant with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=9jnY9PJjYVk">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

The Custom AI Assistant is designed to understand user prompts and engage in dialogue, providing an interactive and user-friendly experience. Harnessing state-of-the-art models for speech recognition and natural language processing (NLP), the application utilizes the OpenVINO™ toolkit to create a streamlined, voice-activated interface that developers can easily integrate and deploy.

This kit uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [Llama 3](https://llama.meta.com/llama3/)

Check out our [AI Reference Kits repository](/) for other kits.

![custom-ai-assistant](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/e0c2f2db-c213-4071-970b-09ebc1eea710)

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Accessing Llama](#how-to-access-llama)
  - [Running the Application](#running-the-application)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Installing Prerequisites

This project requires Python 3.10 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you may also need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)._

## Setting Up Your Environment

### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/ai_ref_kits/custom_ai_assistant
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
## Accessing Llama

_NOTE: If you already have access to the Llama model weights, skip to the authentication step, which is mandatory for converting the Llama model._

### Accessing Original Weights from Meta AI

To access the original Llama model weights:

Visit [Meta AI's website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and fill in your details, including your name, email, and organization.
Accept the terms and submit the form. You will receive an email granting access to download the model weights.

### Using Llama with Hugging Face

Set Up a Hugging Face Account: If you don't have one, create a [Hugging Face account](https://huggingface.co/welcome).

Authenticate with Meta AI: Go to the Llama model (v2 or v3) page on Hugging Face. To authenticate, enter the same email address you used for the Meta AI website. After authentication, you'll gain access to the model.

To use the model, authenticate using the Hugging Face CLI:

```shell
huggingface-cli login
```
When prompted to add the token as a git credential, respond with 'n'. This step ensures that you are logged into the Hugging Face API and ready to download the model.

Now, you're ready to download and optimize the models required to run the application.

## Model Conversion and Optimization

_NOTE: This reference kit requires much bandwidth and disk space (>8GB) for downloading models. Also, the conversion may take much time (>2h with slow networks) and need much memory (>32GB)) when running for the first time as the models used here are huge. After the first run, the subsequent runs will be done much faster._

The application uses three separate models for its operation, each requiring conversion and optimization for use with OpenVINO™. Follow the order below to convert and optimize each model:

1. Automated Speech Recognition Distil-Whisper Conversion:
```shell
python convert_and_optimize_asr.py --asr_model_type distil-whisper-large-v3 --precision int8
```

For GPU, please use the default float16 precision.
```shell
python convert_and_optimize_asr.py --asr_model_type distil-whisper-large-v3
```

This script will convert and optimize the automatic speech recognition (ASR) model performing weight quantization. 

⚠️⚠️ Warning ⚠️⚠️: On Windows you will see an "Permission Error" message due to the export function [bug](https://github.com/openvinotoolkit/openvino_build_deploy/issues/89). The model will be exported successfully, however, you may want to clear the temp directory manually.


If you want to convert speech to text in Chinese, you could choose the Chinese fine-tuned ASR model with the following:
```shell
python convert_and_optimize_asr.py --asr_model_type belle-distilwhisper-large-v2-zh --precision int8
```

For GPU, please use the default float16 precision.
```shell
python convert_and_optimize_asr.py --asr_model_type belle-distilwhisper-large-v2-zh
```

2. Chat LLama Conversion:
   
For desktop or server processors,
```shell
python convert_and_optimize_chat.py --chat_model_type llama3.1-8B --precision int4
```
For AI PC or edge use cases,
```shell
python convert_and_optimize_chat.py --chat_model_type llama3.2-3B --precision int4
```

This script will handle the conversion and optimization of the chat model, performing weights quantization. 
If you want to perform the conversation in Chinese, you could choose the Chinese LLM Qwen2-7B-instruct model with the following:
```shell
python convert_and_optimize_chat.py --chat_model_type qwen2-7B --precision int4
```

After running the conversion scripts, you can run app.py to launch the application.

## Running the Application (Gradio Interface)

_NOTE: This application requires much memory (>16GB) as the models used here are huge (especially chatbot). It may also be slow for less powerful devices._

Execute the `app.py` script with the following command, including all necessary model directory arguments:
```shell
python app.py --asr_model_dir path/to/asr_model --chat_model_dir path/to/chat_model
```
Replace `path/to/asr_model` and `path/to/chat_model` with actual paths to your respective models. Add `--public` to make it publicly accessible.

### Accessing the Web Interface
After running the script, Gradio will provide a local URL, typically `http://127.0.0.1:XXXX`, which you can open in your web browser to start interacting with the assistant. If you configured the application to be accessible publicly, Gradio will also provide a public URL.

Trying Out the Application
1. Navigate to the provided Gradio URL in your web browser.
2. You will see the Gradio interface with options to input voice.
3. To interact using voice:
    - Click on the microphone icon and speak your query.
    - Wait for the assistant to process your speech and respond.
4. The assistant will respond to your query in text.

Feel free to engage with the Custom AI Assistant, ask questions, or give commands as per the assistant's capabilities. This hands-on experience will help you understand the assistant's interactive quality and performance.

Enjoy exploring the capabilities of your Custom AI Assistant!

# Benchmark Results 

<img width="1229" height="982" alt="image" src="https://github.com/user-attachments/assets/5cd865c8-6f67-439f-bb31-d1dc5aa5727b" />

You can evaluate performance benchmarks for models like Llama3.2-3B-Instruct and Qwen2-7B across a range of Intel® platforms using the [OpenVINO™ Model Hub](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html). The Model Hub is a curated resource for developers to explore OpenVINO optimized models and assess their latency and throughput on Intel® CPUs, integrated GPUs, discrete GPUs, and NPUs.

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=ai_ref_kits/custom_ai_assistant&file=README.md" />
