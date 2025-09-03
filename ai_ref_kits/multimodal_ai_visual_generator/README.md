
<div align="center">

# Multimodal AI Visual Generator with OpenVINO™ Toolkit  
Transform a single creative prompt into a vivid illustrated story or artistic T-shirt design using optimized LLMs and text-to-image models.

  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
  </h4>
</div>

[![Apache License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

---

The Multimodal AI Visual Generator is a multimodal generative AI reference kit that demonstrates how large language models (LLMs) and diffusion-based image generation models can work together in a creative pipeline. It allows users to transform a single text prompt into detailed illustrated stories or stylized T-shirt design concepts, using optimized models for local deployment.

By combining LLM-driven prompt generation with image synthesis, the application shows how OpenVINO™ can accelerate multimodal generative AI workflows across Intel® NPUs, CPUs, integrated GPUs, and discrete GPUs. Multimodal AI Visual Generator delivers a complete pipeline, covering prompt input, scene generation, visual rendering, and PDF export.

This kit serves as a practical foundation for building real-world applications in storytelling, branding, education, and other creative domains powered by generative AI.

This kit uses the following technology stack:

- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [OpenVINO™ GenAI](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html)
- [Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html)
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (LLM)
- [FLUX.1](https://github.com/black-forest-labs/flux) (text-to-image)
- [Streamlit](https://docs.streamlit.io/) (frontend)
- [FastAPI](https://fastapi.tiangolo.com/) (backend)

Check out our [AI Reference Kits repository](https://github.com/openvinotoolkit/openvino_build_deploy) for other kits.

![visual-gen-studio](https://github.com/user-attachments/assets/43d6e473-19c1-4047-aee1-07a484cd0dc1)

---

## What's Included

This project includes:

- **LLM-based prompt generation**
- **Text-to-image rendering**
- **Interactive web UI** built with Streamlit
- **Conversion scripts** to optimize models using OpenVINO
- **PDF output** generation

---
### What's New

New updates will be added to this contents list.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Install Prerequisites](#install-prerequisites)
  - [Set Up Your Environment](#set-up-your-environment)
  - [Convert and Optimize the Model](#convert-and-optimize-the-model)
  - [Run the Application](#run-the-application)
-  [Try it Out](#try-it-out) 
- [Additional Resources](#additional-resources)

</details>

## Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

### Install Prerequisites

This project requires Python 3.10 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

- Python ≥ 3.10
- Git and Git LFS
- (Windows only) [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

Install dependencies:

Install libraries and tools:

```bash
sudo apt update
sudo apt install git git-lfs python3-venv python3-dev
git lfs install
```
## Set Up Your Environment

To set up your environment, you first clone the repository, then create a virtual environment, activate the environment, and install the packages.

### Clone the Repository

To clone the repository and navigate into the directory, run the following command:

```bash
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
cd openvino_build_deploy/ai_ref_kits/multimodal_ai_visual_generator
```

### Create a Virtual Environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command to create and activate the environment:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Install Python Dependencies

To install the required packages, run the following commands:

```bash
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

---
### Accessing Gated Models with Hugging Face


Set Up a Hugging Face Account: If you don't have one, create a [Hugging Face account](https://huggingface.co/welcome).

Authenticate gated models on Hugging Face. To authenticate, enter the same email address you used for the Hugging Face website. After authentication, you'll gain access to the model.

To use the model, authenticate using the Hugging Face CLI:

```shell
huggingface-cli login
```
When prompted to add the token as a git credential, respond with 'n'. This step ensures that you are logged into the Hugging Face API and ready to download the model.

Now, you're ready to download and optimize the models required to run the application.

**Note**: Some demonstrated models can require at least 32GB RAM for conversion and running.

Use the provided scripts to export and optimize the models. When you run them, by default, the scripts will prompt you with a numbered list of supported models to choose from interactively.


## Download Pre-optimized Models (Option 1)

Download the pre-optimized LLMs model
```shell
huggingface-cli download OpenVINO/Qwen2.5-7B-Instruct-int4-ov --local-dir models/Qwen2.5-7B-Instruct-int4
```
Download the pre-optimized image generation model
```shell
huggingface-cli download OpenVINO/FLUX.1-schnell-int4-ov --local-dir models/FLUX.1-schnell-int4
```

## Convert and Optimize the Model (Option 2)

### Convert the Chat LLM

```shell
python convert_and_optimize_llm.py
```

### Convert the Image Generation Model

```bash
python convert_and_optimize_text2image.py
```

The script will then handle download, export, and OpenVINO optimization automatically.

Alternatively, you can also run the scripts non-interactively by directly specifying the model and precision as shown below:

```bash
python convert_and_optimize_llm.py --chat_model_type qwen2-7B --precision int4
python convert_and_optimize_text2image.py --image_model_type flux.1-schnell --precision int4
```

---

## Run the Application

This app has two components: a FastAPI backend and a Streamlit frontend.

### Step 1: Run FastAPI (in Terminal 1)

The FastAPI backend can be configured using environment variables to specify which models to use:

- `IMAGE_MODEL_TYPE`: The type of image generation model to use (default: "flux.1-schnell")
- `LLM_MODEL_TYPE`: The type of language model to use (default: "qwen2-7B")
- `MODEL_PRECISION`: The precision to use for both models (default: "int4")

You can set these variables when running the application:

```bash
cd openvino_build_deploy/ai_ref_kits/multimodal_ai_visual_generator
source venv/bin/activate         # On Windows: venv\Scripts\activate

# Run with default values
uvicorn main:app --host 0.0.0.0 --port 8000

# Or run with custom model configuration. e.g., using the pre-optimized models above 
IMAGE_MODEL_TYPE="FLUX.1-schnell" LLM_MODEL_TYPE="Qwen2.5-7B-Instruct" MODEL_PRECISION="int4" uvicorn main:app --host 0.0.0.0 --port 8000
#On Windows PowerShell
$env:IMAGE_MODEL_TYPE="FLUX.1-schnell" 
$env:LLM_MODEL_TYPE="Qwen2.5-7B-Instruct" 
$env:MODEL_PRECISION="int4" 
uvicorn main:app --host 0.0.0.0 --port 8000

```

If no environment variables are set, the application will use the default values.

### Step 2: Run Streamlit UI (in Terminal 2)

```bash
cd openvino_build_deploy/ai_ref_kits/multimodal_ai_visual_generator
source venv/bin/activate         # On Windows: venv\Scripts\activate
streamlit run streamlit_app.py
```

Once both servers are up, the browser will open to `http://localhost:8501`.

---

## Try it Out

Illustration mode:
- "A bunny explores a candy forest"
- "A robot learns to bake cookies"

Branding mode:
- "A turtle with a magic wand"
- "A happy robot with a party hat"

---

# Benchmark Results 

<img width="1367" height="942" alt="image" src="https://github.com/user-attachments/assets/07efd67d-989f-459a-b8b9-7ecebf27abaa" />

You can evaluate performance benchmarks for models like Qwen2.5-7B and FLUX.1-schnell across a range of Intel® platforms using the [OpenVINO™ Model Hub](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html). The Model Hub is a curated resource for developers to explore OpenVINO optimized models and assess their latency and throughput on Intel® CPUs, integrated GPUs, discrete GPUs, and NPUs.

---
## Additional Resources

- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO's documentation](https://docs.openvino.ai/2024/home.html)
- Browse the [OpenVINO™ Model Hub](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html) for real-world model benchmarks

<p align="right"><a href="#top">Back to top ⬆️</a></p>

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca?project=ai_ref_kits/multimodal_ai_visual_generator?file=README.md" />




