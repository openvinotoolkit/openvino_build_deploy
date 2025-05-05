
<div align="center">

# Visual Gen Studio with OpenVINO™ Toolkit  
Transform a single creative prompt into a vivid illustrated story or artistic T-shirt design using optimized LLMs and text-to-image models.

[![Apache License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

</div>

---

## Overview

The Visual Gen Studio is designed to turn user prompts into either a multi-scene illustrated story or visually striking T-shirt design concepts. 
By combining large language models with image generation pipelines, the application demonstrates how OpenVINO™ can accelerate multimodal generative AI workflows across CPU, iGPU, and GPU devices.

This kit uses the following technology stack:

- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [OpenVINO™ GenAI](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html) and [Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html) for model optimization and inference
- Qwen2-7B (LLM), FLUX.1 (text-to-image) - both optimized with OpenVINO
- Streamlit (frontend) and FastAPI (backend)

Check out our [AI Reference Kits repository](https://github.com/openvinotoolkit/openvino_build_deploy) for other kits.

![visual-gen-studio](https://github.com/user-attachments/assets/43d6e473-19c1-4047-aee1-07a484cd0dc1)

---

## Kit Structure

This project includes:

- **LLM-based prompt generation**
- **Text-to-image rendering**
- **Interactive web UI** built with Streamlit
- **Conversion scripts** to optimize models using OpenVINO
- **PDF output** generation

---

## Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

### 1. Prerequisites

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

### 2. Clone the Repository

To clone the repository and navigate into the directory , run the following command:

```bash
git clone https://github.com/AnishaUdayakumar/openvino_build_deploy.git
cd openvino_build_deploy/ai_ref_kits/Visual_Story_Maker
```

### 3. Create a Virtual Environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command to create and activate the environment:

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 4. Install Python Dependencies

To install the required packages, run the following commands:

```bash
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

---

## Model Conversion

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

### Convert the Chat LLM

```shell
python convert_and_optimize_llm.py
```

### Convert the Image Generation Model

```bash
python convert_and_optimize_texttoimage.py
```

The script will then handle download, export, and OpenVINO optimization automatically.

Alternatively, you can also run the scripts non-interactively by directly specifying the model and precision as shown below:

```bash
python convert_and_optimize_llm.py --chat_model_type qwen2-7B --precision int4
python convert_and_optimize_texttoimage.py --image_model_type flux.1-schnell --precision int4
```

---

## Run the Application

This app has two components: a FastAPI backend and a Streamlit frontend.

### Step 1: Run FastAPI (in Terminal 1)

```bash
cd openvino_build_deploy/ai_ref_kits/Visual_Story_Maker
source venv/bin/activate         # On Windows: venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Step 2: Run Streamlit UI (in Terminal 2)

```bash
cd openvino_build_deploy/ai_ref_kits/Visual_Story_Maker
source venv/bin/activate         # On Windows: venv\Scripts\activate
streamlit run streamlit_app.py
```

Once both servers are up, the browser will open to `http://localhost:8501`.

---

## Example Prompts

Illustration mode:
- "A bunny explores a candy forest"
- "A robot learns to bake cookies"

Branding mode:
- "A turtle with a magic wand"
- "A happy robot with a party hat"

---

## Additional Resources

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Git LFS](https://git-lfs.com/)
