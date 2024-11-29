<div id="top" align="center">
  <h1>Conversational AI Chatbot with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

The Conversational AI Chatbot is an open-source, voice-driven chat agent that answers spoken questions with meaningful, spoken responses.  It can be configured to respond in any type of scenario or context. This kit demonstrates the AI Chatbot’s capabilities by simulating the experience of talking to a hotel concierge.

Developers can easily integrate and deploy the AI Chatbot for their applications. It harnesses models for speech recognition and text-to-speech (TTS) synthesis. The Chatbot is configured to understand user prompts, engage in meaningful dialogue, and provide spoken, conversational responses. It uses Intel OpenVINO™, which is a toolkit that enables developers to deploy deep learning models on hardware platforms.


This kit uses the following technology stack:
- [OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([Docs](https://docs.openvino.ai/))
- [Meta’s Llama 3.2](https://llama.meta.com/llama3/)
- [OpenAI Whisper](https://openai.com/index/whisper/)
- [MeloTTS](https://github.com/myshell-ai/MeloTTS/tree/main)
- [Gradio interface](https://www.gradio.app/docs/gradio/chatinterface)

For other Intel AI kits, see the [Edge AI Reference Kits repository](/).

![conversational-ai-chatbot](https://github.com/user-attachments/assets/262ba878-b845-445b-aff7-0a118517d409)

### What's New

New updates will be added to this contents list.

<details open><summary><b>Table of Contents</b></summary>
  
- [Get Started](#get-started)
  - [Install Prerequisites](#install-prerequisites)
  - [Setg Up Your Environment](#set-up-your-environment)
  - [Get Access to Llama](#get-access-to-llama)
  - [Convert and Optimize the Model](#convert-and-optimize-the-model)
  - [Run the Application](#run-the-application-gradio-interface)
- [Additional Resources](#additional-resources)

</details>

# Get Started

To get started with the Conversational AI Chatbot, you install Python, set up your environment, and then you can run the application.

## Install Prerequisites

This project requires Python 3.8 or higher and a few libraries. If you don't already have Python installed on your machine, go to [https://www.python.org/downloads/](https://www.python.org/downloads/) and download the latest version for your operating system. Follow the prompts to install Python, and make  sure to select  the option to add Python to your PATH environment variable.

To install the Python libraries and tools, run this command:

```shell
sudo apt install git gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you might also have to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)._

## Set Up Your Environment

To set up your environment, you first clone the repository, then create a virtual environment, activate the environment, and install the packages.

### Clone the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

This command  clones the repository into a directory named "openvino_build_deploy" in the current directory. After the directory is cloned, run the following command to go to that directory:

```shell
cd openvino_build_deploy/ai_ref_kits/conversational_ai_chatbot
```

### Create a Virtual Environment

To create a virtual environment, open your terminal or command prompt, and go to the directory where you want to create the environment. 

Run the following command:

```shell
python3 -m venv venv
```
This creates a new virtual environment named "venv" in the current directory.

### Activate the Environment

The command you run to activate the virtual environment you created depends on whether you have a Unix-based operating system (Linux or macOS) or a Windows operating system.

To activate  the virtual environment for a **Unix-based** operating system, run:

```shell
source venv/bin/activate   # This command is for Unix-based operating systems such as Linux or macOS
```

To activate the virtual environment for a **Windows** operating system, run:

```shell
source venv\Scripts\activate   # This command is for Windows operating systems
```
This activates the virtual environment and changes your shell's prompt to indicate that you are now working in that environment.

### Install the Packages

MeloTTS is a high-quality multilingual text-to-speech library by MIT and MyShell.ai. However, the installation of this model's dependencies needs to separated from the rest of the dependency installation process, due to some potential conflict issues. Details of this model could be found [here](https://github.com/myshell-ai/MeloTTS). Using the following command to install MeloTTS locally.

```shell
python -m pip install --upgrade pip 
pip install git+https://github.com/myshell-ai/MeloTTS.git@5b538481e24e0d578955be32a95d88fcbde26dc8 --no-deps
python -m unidic download
```

To install the other packages, run the following commands:

```shell 
pip install -r requirements.txt
```

## Get Access to Llama

_NOTE: If you already have access to the Llama model weights, you can proceed to the authentication step, which is mandatory to convert the Llama model._

## Convert and Optimize the Model

The application uses three separate models. Each model requires conversion and optimization for use with OpenVINO™. The following process includes a step to convert and optimize each model.

_NOTE: This reference kit requires more than 8GB of bandwidth and disk space for downloading models. Because of the large model size, when you run the kit for the first time, the conversion can take more than two hours and require more than 32GB of memory. After the first run, the subsequent runs should finish much faster._


### Step 1. Automated Speech Recognition Distil-Whisper Conversion  

The ASR model converts spoken language (audio) to written text. This functionality is crucial because it enables the chatbot to understand and process voice-based user queries.

To convert and optimize the automatic speech recognition (ASR) model performing weights quantization, run:
```shell
python convert_and_optimize_asr.py --asr_model_type distil-whisper-large-v3 --precision int8
```

### Step 2. Chat Model, Embedding Model, and Reranker Model Conversion
  
The _chat model_ is the core of the chatbot's ability to generate meaningful and context-aware responses. It processes the text input from the ASR model and produces a human-like response.  

The _embedding model_ converts text data (both user queries and potential responses or knowledge base entries) to numerical vectors. These vectors are essential for tasks such as semantic search and similarity matching.

The _reranker model_ is used in retrieval-augmented generation (RAG) configurations to reorder or _rerank_ retrieved results, to make sure that the most relevant information is presented to the user.

This conversion script handles the conversion and optimization of:

- The chat model (`llama3.2-3B`) with `int4` precision.
- The embedding model (`bge-large`) with `FP32` precision.
- The reranker model (`bge-reranker-large`) with `FP32` precision.

Before you can run the script to convert the models, you must have a Hugging Face token (`--hf_token`) for authentication, which allows you to get access to gated models, such as Llama. After the models are converted, they’re saved to the model directory you specify when you run the script.

To get access to the original Llama model weights:
1. Go to the Llama model page on Hugging Face [meta-llama/Meta-Llama 3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).  
_NOTE: These instructions specify Llama 3.2-3B, but the default version is Llama 3.1._
2. Read and accept the license agreement.  
_Requests can take up to one hour to process._

After you get access to the Llama model weights, you can convert the chat, embedding, and reranker models.

To convert the chat, embedding, and reranker models, run:
```shell
python convert_and_optimize_chat.py --chat_model_type llama3.2-3B --embedding_model_type bge-large --reranker_model_type bge-reranker-large --precision int4 --hf_token your_huggingface_token --model_dir model
```

### Step 3. Text-to-Speech (TTS) Model Conversion

The text-to-speech (TTS) model converts the chatbot's text responses to spoken words, which enables voice output. The application uses MeloTTS model for TTS. The TTS model doesn't require conversion. They are compiled at runtime using ``torch.compile`` with the OpenVINO backend.

After you run the conversion scripts, you can run app.py to launch the application.

## Run the Application (Gradio Interface)

To run the Conversational AI Chatbot application, you execute the following python script. Make sure to include all of the necessary model directory arguments. 

_NOTE: This application requires more than 16GB of memory because the models are very large (especially the chatbot model). If you have a less powerful device, the application might also run slowly._

For the python script, you must include the following model directory arguments.

- `--asr_model path/to/asr_model`: The path to your ASR (Automatic Speech Recognition) model directory, which uses `int8` precision (for example,  `model/distil-whisper-large-v3-int8`) for efficient speech recognition.

- `--chat_model path/to/chat_model`: The path to your chat model directory (for example, `model/llama3.2-3B-INT4`) that drives conversation flow and response generation.

- `--embedding_model path/to/embedding_model`: The path to your embedding model directory (for example, `model/bge-small-FP32`) for understanding and matching text inputs.

- `--reranker_model path/to/reranker_model`: The path to your reranker model directory (for example, `model/bge-reranker-large-FP32`). This model reranks responses to ensure relevance and accuracy.

- `--personality path/to/personality.yaml`: The path to your custom personality YAML file (for example, `concierge_personality.yaml`).  
This file defines the assistant's personality, including instructions, system configuration, and greeting prompts. You can create and specify your own custom personality file.

- `--example_pdf path/to/personality.yaml`: The path to your custom PDF file which is an additional context (for example, `Grand_Azure_Resort_Spa_Full_Guide.pdf`).  
This file defines the knowledge of the resort in this concierge use case. You can use your own custom file to build a local knowledge base.

- `--public`: Include this flag to make the Gradio interface publicly accessible over the network. Without this flag, the interface is only available on your local machine.

To run the application, execute the `app.py` script with the following command. Make sure to include all necessary model directory arguments.
```shell
python app.py \
  --asr_model path/to/asr_model \
  --chat_model path/to/chat_model \
  --embedding_model path/to/embedding_model \
  --reranker_model path/to/reranker_model \
  --personality concierge_personality.yaml \
  --example_pdf Grand_Azure_Resort_Spa_Full_Guide.pdf \
  --public
```

### Create a Custom YAML Personality File

You can create a personality file for your virtual AI assistant using YAML. Each personality can be customized based on the specific role of the assistant, such as a concierge, bartender, or legal assistant. 

#### Components of a Personality File

A typical YAML personality file has the following sections:

1. **Instructions.** A brief, descriptive title for the assistant.
2. **System Configuration.** Instructions that define the assistant's behavior and limitations.
3. **Greet the User Prompt.** The first interaction when the assistant introduces itself.

#### Tips for Creating the YAML File 

The YAML file _instructions_ section should provide an introduction to the assistant, the title of the assistant, and important notes for the user. It should be clear and concise, and give users context for how to interact with the assistant.

```yaml
instructions: | 
  # [Assistant Name]: [Brief Role Description]

        Instructions for use:  
        1. Provide a brief, step-by-step guide for how the assistant works.  
        2. Include key points that users should know before they interact with the assistant.  
        3. Mention any important disclaimers, if applicable.

        **Note: [Add a disclaimer or key note about what the assistant can and cannot do].**
```

### Use the Web Interface
After the script runs, Gradio provides a local URL (typically `http://127.0.0.1:XXXX`) that you can open in your web browser to interact with the assistant. If you configured the application to be accessible publicly, Gradio also provides a public URL.

#### Test the Application

When you test the AI Chatbot assistant, you can test both the voice interaction and text interaction capabilities of the application.

1. Open a web browser and go to the Gradio-provided URL.  
    _For example, `http://127.0.0.1:XXXX`._
2. Upload a file for RAG context.
    - Use the file upload widget to select and upload your file.
    - Choose a PDF or TXT file for the chatbot to use as a knowledge base.  
      _For example, "Grand_Azure_Resort_Spa_Full_Guide.pdf"._    
3. Test voice interaction with the chatbot.
    - Click the microphone icon and speak your question.
    - Wait for the assistant to process your speech and respond.
4. Test text interaction with the chatbot.
    - Type your question in the text box.
    - To send your question to the chatbot, click **Submit** or press **Enter**.  
    _The assistant responds to your question in text and audio form._

For further testing of the Conversational AI Chatbot, you can engage with the chatbot assistant by asking it questions, or giving it commands that align with the assistant's capabilities. This hands-on experience can help you to understand the assistant's interactive quality and performance.

Enjoy exploring the capabilities of your Conversational AI Chatbot!

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>
