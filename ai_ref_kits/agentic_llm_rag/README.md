<div id="top" align="center">
  <h1>AI Insight Agent with RAG</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;</a>
    <!-- <a href="">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video</a> -->
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

<p align="center">
  <img src="https://github.com/user-attachments/assets/dd626685-7aa6-4e67-a929-5e9be2982800" width="500">
</p>

The AI Insight Agent with RAG uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to interpret user prompts, engage in meaningful dialogue, perform calculations, use RAG techniques to improve its knowledge and interact with the user to add items to a virtual shopping cart. This solution uses the OpenVINO™ toolkit to power the AI models at the edge. Designed for both consumers and employees, it functions as a smart, personalized retail assistant, offering an interactive and user-friendly experience similar to an advanced digital kiosk.

This kit uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen)
- [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Gradio interface](https://www.gradio.app/docs/gradio/chatinterface)

Check out our [AI Reference Kits repository](/) for other kits.

![ai-insight-agent-with-rag](https://github.com/user-attachments/assets/da97bea7-29e8-497f-b7ba-4e00c79773f1)

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#get-started)
  - [Installing Prerequisites](#install-prerequisites)
  - [Setting Up Your Environment](#set-up-your-environment)  
  - [Converting and Optimizing the Model](*convert-and-optimize-the-model)
  - [Running the Application](#run-the-application)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

To get started with the AI Insight Agent with RAG, you install Python, set up your environment, and then you can run the application. We recommend using Ubuntu 24.04 to set up and run this project.

## Installing Prerequisites

This project requires Python 3.8 or higher and a few libraries. If you don't already have Python installed on your machine, go to [https://www.python.org/downloads/](https://www.python.org/downloads/) and download the latest version for your operating system. Follow the prompts to install Python, and make  sure to select the option to add Python to your PATH environment variable.

To install the Python libraries and tools, run this command:

```shell
sudo apt install git gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you might also have to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)._

## Setting Up Your Environment

To set up your environment, you first clone the repository, then create a virtual environment, activate the environment, and install the packages.

### Clone the Repository

To clone the repository, run this command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

This command clones the repository into a directory named "openvino_build_deploy" in the current directory. After the directory is cloned, run the following command to go to that directory:


```shell
cd openvino_build_deploy/ai_ref_kits/agentic_llm_rag
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
source venv/bin/activate   # For Unix-based operating systems such as Linux or macOS
```

To activate the virtual environment for a **Windows** operating system, run:

```shell
venv\Scripts\activate  # This command is for Windows operating systems
```
This activates the virtual environment and changes your shell's prompt to indicate that you are now working in that environment.

### Install the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```
## Converting and Optimizing the Model

The application uses 2 separate models. Each model requires conversion and optimization for use with OpenVINO™. The following process includes a step to convert and optimize each model.

_NOTE: This reference kit requires more than 8GB of bandwidth and disk space for downloading models. Because of the large model size, when you run the kit for the first time, the conversion can take more than two hours and require more than  32GB of memory. After the first run, the subsequent runs should finish much faster._

## Chat Model and Embedding Model Conversion
  
The _chat model_ is the core of the chatbot's ability to generate meaningful and context-aware responses.

The _embedding model_ represents text data (both user queries and potential responses or knowledge base entries) as numerical vectors. These vectors are essential for tasks such as semantic search and similarity matching.

This conversion script handles the conversion and optimization of:

- The chat model (`qwen2-7B`) with `int4` precision.
- The embedding model (`bge-large`) with `FP32` precision.

After the models are converted, they’re saved to the model directory you specify when you run the script.

_Requests can take up to one hour to process._

To convert the chat and embedding models, run:
```shell
python convert_and_optimize_llm.py --chat_model_type qwen2-7B --embedding_model_type bge-large --precision int4 --model_dir model
```

After you run the conversion scripts, you can run `app.py` to launch the application.

## Running the Application (Gradio Interface)

To run the AI Insight Agent with RAG application, you execute the following python script. Make sure to include all of the necessary model directory arguments. 

_NOTE: This application requires more than 16GB of memory because the models are very large (especially the chatbot model). If you have a less powerful device, the application might also run slowly._

After that, you should be able to run the application with default values:

```shell
python app.py
```

For more settings, you can change the argument values:

- `--chat_model`: The path to your chat model directory (for example, `model/qwen2-7B-INT4`) that drives conversation flow and response generation.

- `--rag_pdf`: The path to the document (for example, `data/test_painting_llm_rag.pdf`) that contains additional knowledge for Retrieval-Augmented Generation (RAG).

- `--embedding_model`: The path to your embedding model directory (for example, `model/bge-small-FP32`) for understanding and matching text inputs.

- `--device`: Include this flag to select the inference device for both models. (for example, `CPU`). If you have access to a dedicated GPU (ARC, Flex), you can change the value to `GPU.1`. Possible values: `CPU,GPU,GPU.1,NPU`

- `--public`: Include this flag to make the Gradio interface publicly accessible over the network. Without this flag, the interface will only be available on your local machine.

To run the application, execute the `app.py` script with the following command. Make sure to include all necessary model directory arguments.
```shell
python app.py \ 
  --chat_model model/qwen2-7B-INT4 \
  --embedding_model data/test_painting_llm_rag.pdf \
  --rag_pdf model/bge-small-FP32 \  
  --device GPU.1 \
  --public
```

### System Prompt Usage in LlamaIndex ReActAgent

The LlamaIndex ReActAgent library relies on a default system prompt that provides essential instructions to the LLM for correctly interacting with available tools. This prompt is fundamental for enabling both tool usage and RAG (Retrieval-Augmented Generation) queries.

#### Important:
Do not override or modify the default system prompt. Altering it may prevent the LLM from using the tools or executing RAG queries properly.

#### Customizing the Prompt:
If you need to add extra rules or custom behavior, modify the Additional Rules section located in the system_prompt.py file.

### Use the Web Interface
After the script runs, Gradio provides a local URL (typically `http://127.0.0.1:XXXX`) that you can open in your web browser to interact with the assistant. If you configured the application to be accessible publicly, Gradio also provides a public URL.

#### Test the Application
When you test the AI Insight Agent with RAG application, you can test both the interaction with the agent and the product selection capabilities.

1. Open a web browers and go to the Gradio-provided URL.  
  _For example, `http://127.0.0.1:XXXX`._
2. Test text interaction with the application.  
  - Type your question in the text box and press **Enter**.
  _The assistant responds to your question in text form._

For further testing of the AI Insight Agent with RAG appplication, you can engage with the chatbot assistant by asking it questions, or giving it commands that align with the assistant's capabilities. This hands-on experience can help you to understand the assistant's interactive quality and performance.

Enjoy exploring the capabilities of your AI Insight Agent with RAG appplication!

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>
