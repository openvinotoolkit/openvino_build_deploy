# Bringing Adventure Gaming to Life 🧙 on AI PC with the OpenVINO™ toolkit 💻

**Authors:** Arisha Kumar, Garth Long, Ria Cheruvu, Dmitriy Pastushenkov, Paula Ramos, Raymond Lo, Zhuo Wu

**Contact (for questions):** Ria Cheruvu, Dmitriy Pastushenkov

**Tested on:** Intel® Core™ Ultra 7 and 9 Processors

## Pipeline
![SIGGRAPH Drawing](https://github.com/user-attachments/assets/3ce58b50-4ee9-4dae-aeb6-0af5368a3ddd)

## Installation

1. Clone this repository to get started

2. Download and optimize required models
	- Nano-Llava (MultiModal) - Image Recognition/Captioning from Webcam 
	- Whisper - Speech Recognition
	- Llama3-8b-instruct - Prompt Refinement
	- AI Superesolution - Increase res of generated image
	- Latent Consistency Models - Generating Image
  	- Depth Anything v2 - Create 3d parallax animations
    
	```
    	python -m venv model_installation_venv
	model_installation_venv\Scripts\activate
	pip install -r python3.12_requirements_model_installation.txt
	python download_and_prepare_models.py
    ``` 
	After model installation, you can remove the virtual environment as it isn't needed anymore.


3. Create a virtual env and install the required python packages. Your requirements.txt file will depend on the Python version you're using (3.11 or 3.12) <br>
    ```
    	python -m venv dnd_env
	dnd_env\Scripts\activate
	pip install -r requirements.txt 
	pip install "openai-whisper==20231117" --extra-index-url https://download.pytorch.org/whl/cpu

    ``` 
4. To interact with the animated GIF outputs, you will need to host a simple web server on your system as the final output a player will see. To do so, please install Node.js via [its Download page](https://nodejs.org/en/download/package-manager) and [http-server](https://www.npmjs.com/package/http-server).

Run the following command to start an HTTP server within the repository. You can customize index.html with any additional elements as you'd like.
```
http-server
``` 
5. Open a terminal or you can use the existing one with dnd_env environment activated and start the Gradio GUI - <br>
```
python gradio_ui.py 
```
Click on the web link to open the GUI in the web browser <br>

## How to Use 🛣️
<img width="1270" alt="quick_demo_screenshot" src="https://github.com/user-attachments/assets/ddfea7f0-3f1d-4d1c-b356-3bc959a23837">

### (Step 1 📷) Take a picture
Take a picture via the Gradio image interface of any object you want! Your "theme" will become the image description, if the object in the image is clearly captured.
### (Step 2 🗣️) Speak your prompt
Start the recording, and wait till the server is listening to begin speaking your prompt to life. Click the Stop button to stop the generation.
### (Step 3 ➕) Add theme to prompt
Now, your prompt is transcribed! Click on the "Add Theme to Prompt" button to combine your prompt and theme.
### (Step 4 ⚙️) Refine it with an LLM
You can optionally ask an LLM model to refine your model by clicking on the LLM button. It will try its best to generate a prompt infusing the elements (although it does hallucinate at times).
### (Step 5 🖼️) Generate your image (and depth map)
Click "Generate Image" to see your image come to life! A depth map will automatically be generated for the image as well. Feel free to adjust the advanced parameters to control the image generation model!
### (Step 6 🪄🖼️) Interact with the animated GIF
To interact with the 3D hoverable animation created with depth maps, start a HTTP server as explained above, and you will be able to interact with the parallax.

**Optionally:** Navigate over to *Advanced Parameters*, and set OCR to true and roll a die! 🎲 Take a snapshot using the Gradio Image interface. After recognizing the die, the system will try to output the correct die value, and a special location associated with the number you rolled (see locations.json for the list), to add a theme to your prompts. You can change this and the corresponding theme it sets.
