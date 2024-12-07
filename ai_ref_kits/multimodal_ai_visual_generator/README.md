<div id="top" align="center">
  <h1>Multimodal AI Visual Generator with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=kn1jZ2nLFMY">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video&nbsp;·</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

The Multimodal AI Visual Generator is designed for rapid prototyping, instant iteration, and seamless visualization of complex concepts. The kit integrates image creation with generative AI, automatic speech recognition (ASR), speech synthesis, large language models (LLMs), and natural language processing (NLP). It processes multimodal inputs from sources such as cameras, voice commands, or typed text to generate AI-driven visual outputs. Utilizing the Intel OpenVINO™ toolkit, the system enables seamless deployment of deep learning models across hardware platforms. Explore the demo to see its real-time visual generative AI workflow in action.

This kit uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [nanoLLaVA (multimodal)](https://huggingface.co/qnguyen3/nanoLLaVA)
- [Whisper](https://github.com/openai/whisper)
- [Llama3-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Single Image Super Resolution](https://arxiv.org/abs/1807.06779)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

Check out our [AI Reference Kits repository](/) for other kits.

![kit-gif](https://github.com/user-attachments/assets/f113a126-4b44-4488-be4e-e4bf52a6cebc)

Contributors: Ria Cheruvu, Garth Long, Arisha Kumar, Paula Ramos, Dmitriy Pastushenkov, Zhuo Wu, and Raymond Lo.

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Running the Application](#running-the-application)
- [Additional Resources](#additional-resources)

</details>

# Getting Started
Now, let's dive into the steps starting with installing Python. 

## Installing Prerequisites

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project. This project requires Python 3.8 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git git-lfs gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you will probably need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Setting Up Your Environment
### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/ai_ref_kits/multimodal_ai_visual_generator
```

Next, you’ll download and optimize the required models. This will involve the creation of a temporary virtual environment and the running of a download script. Your requirements.txt file will depend on the Python version you're using (3.11 or 3.12).

- nanoLLaVA (multimodal): Image recognition/captioning from webcam 
- Whisper: Speech recognition
- Llama3-8b-instruct: Prompt refinement
- Latent Consistency Models: Image generation
  
**Note:** If you would like to run Latent Consistency Models on the NPU, as shown in the demo above, please follow the following steps: Download the model from this location "https://huggingface.co/Intel/sd-1.5-lcm-openvino" and compile it via the steps located at https://github.com/intel/openvino-ai-plugins-gimp/blob/v2.99-R3-staging/model_setup.py. 

- AI Super Resolution: Increase the resolution of the generated image
- Depth Anything v2: Create 3d parallax animations
    
```shell
python3 -m venv model_installation_venv
source model_installation_venv/bin/activate
pip install -r python3.12_requirements_model_installation.txt
python3 download_and_prepare_models.py
``` 
After model installation, you can remove the `model_installation_venv` virtual environment as it is no longer needed.

### Creating a Virtual Environment

To create a virtual environment, open your terminal or command prompt and navigate to the directory where you want to create the environment. Then, run the following command:

```shell
python3 -m dnd_env
```
This will create a new virtual environment named "dnd_env" in the current directory.

### Activating the Environment

Activate the virtual environment using the following command:

```shell
source dnd_env/bin/activate   # For Unix-based operating systems such as Linux or macOS
```

_NOTE: If you are using Windows, use the `dnd_env\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

### Installing the Packages

To install the required packages, run the following commands:

```shell
pip install -r requirements.txt 
pip install "openai-whisper==20231117" --extra-index-url https://download.pytorch.org/whl/cpu
``` 

## Running the Application
![SIGGRAPH Drawing](https://github.com/user-attachments/assets/3ce58b50-4ee9-4dae-aeb6-0af5368a3ddd)

To interact with the animated GIF outputs, host a simple web server on your system as the final output. To do so, please install Node.js via [its Download page](https://nodejs.org/en/download/package-manager) and [http-server](https://www.npmjs.com/package/http-server).

Run the following command to start an HTTP server within the repository. You can customize index.html with any additional elements you'd like.

```shell
http-server -c10
``` 

Open a terminal or you can use the existing one with `dnd_env` environment activated and start the Gradio GUI - <br>

```shell
python3 gradio_ui.py 
```

Click on the web link to open the GUI in the web browser.

![demo screenshot](https://github.com/user-attachments/assets/ddfea7f0-3f1d-4d1c-b356-3bc959a23837)

### 📷 Submit a picture
Take or upload a picture of any object via the Gradio image interface. Your "theme" will become the image description, if the object in the image is clearly captured.

### 🗣 Speak your prompt
Start or upload a recording, wait for the server to listen, and speak your prompt to life. Click the “Stop” button to stop the generation.

### ➕ Add a theme to prompt
Now, your prompt is transcribed! Click the "Add Theme to Prompt" button to combine your prompt and theme.

### ⚙️ Refine it with an LLM
You can optionally ask an LLM model to refine your model by clicking the LLM button. It will try its best to generate a prompt infusing the elements.

### 🖼️ Generate your image and depth map
Click "Generate Image" to see your image come to life. A depth map will automatically be generated for the image as well. Feel free to adjust the advanced parameters to control the image generation model.

### 🪄🖼️ Interact with the animated GIF
To interact with the 3D hoverable animation created with depth maps, start an HTTP server as explained above, and you will be able to interact with the parallax.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>
