<div id="top" align="center">
  <h1>Automated Self-Checkout with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="self-checkout-recipe.ipynb">📔&nbsp;Jupyter&nbsp;Notebook&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=VrJRr_thxcs">📺&nbsp;Overview&nbsp;Video&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=rw8cTr-hD-g">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video&nbsp;·</a>
    <a href="https://www.intel.com/content/www/us/en/developer/articles/training/detect-objects-openvino-automated-self-checkout.html">📚&nbsp;Step&#8209;by&#8209;step&nbsp;Tutorial</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

Automated Self-Checkout is designed to help automate checkout for retail businesses by analyzing video streams and detecting and tracking interactions with retail products. It uses OpenVINO™, a toolkit that enables developers to deploy deep learning models on various hardware platforms.

This kit uses the following technology stack:

- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [Ultralytic YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Supervision](https://supervision.roboflow.com/latest/)
- [Gradio](https://www.gradio.app/)

Check out our [AI Reference Kits repository](/) for other kits.

![automated-self-checkout](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/965a6604-fa15-427e-9d44-c23fa0bbeb6b)
**Figure 0. Automated Self-Checkout. Expected Result.**

### What's New

[2025-02-26] Gradio-based Demo UI and Scripts.

- It incorporates a new Gradio-based UI that provides interactivity and additional options to view, process, and review the video ([See the Jupyter Notebook](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/automated_self_checkout/self-checkout-recipe.ipynb)), detections/tracking, and log messages.
- It adds a set of scripts to make it easier to test the demo. For example, you can install, run, or clean the environment and run the demo visually directly (See the ['Setup Scripts'](#automated-installation-running-and-cleaning) section).

<details open><summary><b>Table of Contents</b></summary>

- [Getting Started](#getting-started)
  - [Automated Installation, Running, and Cleaning](#automated-installation-running-and-cleaning)
  - [Manual - Installing Prerequisites](#manual---installing-prerequisites)
  - [Manual - Setting Up Your Environment](#manual---setting-up-your-environment)
  - [Manual - Running the Application](#manual---setting-up-your-environment)
- [Benchmarking the Model with OpenVINO's `Benchmark_App`](#benchmarking-the-model-with-openvinos-benchmark_app)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Automated Installation, Running, and Cleaning

A set of scripts provides a quick run to install, run, and clean the environment and run the demo visually in a direct way. It targets Ubuntu Linux 22.04+. On the other hand, you could follow the [next steps](#manual---installing-prerequisites) for a manual and step-by-step installation in case of a different platform.

| Script | Objective |
|:---|:---|
|**setup.sh**|It prepares and installs all required components and a virtual environment to run the JupyterLab server, opening the Automated Self-checkout notebook.|
|**runEnv.sh**|Once installed, it starts the JupyterLab server to continue working in the Tutorial. It starts from the same point where the server was before shut down.|
|**runDemo.sh**|It shows a Gradio-based UI (It executes the tutorial steps in the background) that provides interaction and additional features to explore object detection, tracking algorithm, and UC capacities.|
|**cleanEnv.sh**|It recursively cleans the virtual environment, downloaded model, and the rest of the information under the folder of this reference implementation. It keeps the scripts only in case users wish to reinstall and run the demo again.|

### Environment Installation

It creates a folder in the home directory to download the repository and initialize the virtual environment containing all required libraries to run the demo or use the jupyter lab notebook.

1. Download [setup.sh](./setup/setup.sh) in your home directory
2. Grant execution permissions
  
    ```bash
    sudo chmod +x setup.sh
    ```

3. Run the setup file.

    ```bash
    ./setup.sh
    ```

Once finished the environment setup and installation, jupyter lab is automatically opened in your browser (See Figure 1).

![The Setup Script Output and Jupyter Lab Welcome Screen](https://github.com/user-attachments/assets/2073b0ec-e3b3-4d29-a6f8-12fbf5180fb3)
**Figure 1. The Setup Script Output and Jupyter Lab Welcome Screen**

> When Jupyter Lab is not automatically open on the browser, you can click on the link in the yellow rectangle.

The JupyterLab notebook is ready to be executed, and all required libraries are installed. You can run the instructions step by step to follow the Tutorial. However, if you want to follow the Tutorial at a later time, you can shut down the Jupyter Lab server in the console using the interruption keystroke ("Ctrl + C"). As shown in Figure 2, the server will ask for confirmation (See yellow circles), after which, once received, the server finishes, and the virtual environment gets deactivated.

![Finishing the JupyterLab Server](https://github.com/user-attachments/assets/de472eea-e81b-418f-9393-e2c7c3b3453d)
**Figure 2. Finishing the JupyterLab Server**

You can start the tutorial anytime by going to the oneclickai directory and running the runEnv.sh script, as shown in Figure 3.

![Restarting the JupyterLab Server](https://github.com/user-attachments/assets/d527c824-9436-4a64-9b73-2ab588a0f8ae)
**Figure 3. Restarting the JupyterLab Server**

As you may know, the notebook allows you to experiment with the different steps and run each per time until reaching the final output. Again, "Ctrl + C" in the console lets you shut down the server.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

### Direct Running of the Demo

The demo incorporates a new Gradio-based UI (More about [Gradio](https://www.gradio.app/)) providing interactivity and additional options to process the video ([See the Jupyter Notebook](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/automated_self_checkout/self-checkout-recipe.ipynb)). Some of the main features are:

- It can choose a different video for previewing (full or small screen) and processing it.
- The processed video is shown when the batch of frames is released (progressive).
- It can save the final processed video for future reference.
- A Log table allows the user to filter outputs dynamically while the video is under processing or once processed.
- A bar plot shows the detected object classes and their frequency.
- The purchased items are shown as an independent table to contrast them with log messages and detected objects.

The following simple steps are all you need to start the demo:

1. Go to the  ~/oneclickai folder
2. Run the runDemo.sh script. The YOLO model will be automatically downloaded in case of required. You will see a similar to Figure 4.

![Starting the Demo UI](https://github.com/user-attachments/assets/352b9b7b-ba50-43b8-beb3-b81139facefe)
**Figure 4. Starting the Demo UI**

The demo UI has 4 main regions as follows (See Figure 5):

- **Video Source (Top Left component)**: It initializes with a default video for your reference. You can pre-visualize the video in the current window or using full-screen mode. If you want to use it and see the demo running, just press the "Start video processing" button. However, you could change the video by pressing the "x" in the top right section of it and choosing a new one. The mp4 format is supported.
- **Processed Video (Top right component)**: This component shows the processed video once it has been released by the detection and tracking algorithm. It updates the frames as soon as they are available, and it does not require waiting for the whole video processing. Thus, it can see partial results according to the video processing advances. You can download the final processed video by clicking the &#10515; icon at the top right section of the video.
- **Detection Message Log (Bottom left component)**: It is a table that incorporates the actions informed (i.e., add or remove) for the different classes (for example, banana) and instances (#29 banana) with the corresponding timestamp. Recent logs are included at the beginning of the table, leaving the older ones below. The search text field allows you to filter the table according to the specified text. It is updated as soon as the algorithm releases the messages.
- **Detected Components and Purchased Items (Bottom right components)**: Detected components is a barplot describing the detected classID and the associated detection frequency. It is updated with video processing, and the figure can be saved as an image file (PNG). On the other hand, the "Purchased Items" table contains the output understood by the algorithm as purchased effectively after the adding and removing operations. It is updated with the video processing.

![Final State for the Demo UI using the Default Video](https://github.com/user-attachments/assets/0030a106-6005-4136-8ff4-2167472526b5)
**Figure 5. Final State for the Demo UI using the Default Video**

You can filter the log table, save the processed video, and save the bar plot as an image even once finished the video processing of your video. It helps track objects through the video and review those strange situations, for example, detections of a "carrot" or "sports ball."

The source code of this demo is included in the [self-checkout-recipe.ipynb](self-checkout-recipe.ipynb) file to review the step-by-step approach or in the [directrun.py](./directrun.py) file for a direct execution. You can finish the server using the "Ctrl + C" keystroke.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

### Cleaning the environment

It removes the OpenVINO framework, virtual environment, and associated libraries, leaving the oneclickai folder with the setup scripts. It does not remove the Linux packages (for example, git). You can remove the folder with a simple deletion or use setup scripts to reinstall and run the demo when required.

![Cleaning the Environment](https://github.com/user-attachments/assets/5618339d-682f-474a-b988-c37cc487f30b)
**Figure 6. Cleaning the Environment**

As the previous figure shows (See Figure 6), the script requests confirmation to proceed with the directory removal. Once done, the scripts are the only remaining files. You can delete the folder if it is no longer required because all the required codes are installed exclusively in the "oneclickai "folder without external ramifications in the filesystem. The idea is to have as simple an installation and removal approach as possible.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

## Manual - Installing Prerequisites

This project requires Python 3.9 or higher and a few libraries. If you don't have Python installed on your machine, go to <https://www.python.org/downloads/> and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git git-lfs gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you will probably need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Manual - Setting Up Your Environment

### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/ai_ref_kits/automated_self_checkout
```

Then pull the video sample:

```shell
git lfs -X= -I=data/ pull
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

## Manual - Running the Application

You can run [self-checkout-recipe.ipynb](self-checkout-recipe.ipynb) to learn more about the inference process.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

# Benchmarking the Model with OpenVINO's `benchmark_app`

Benchmarking provides insight into your model's real-world performance. Performance may vary based on use and configuration.

## Benchmark Results

![YOLOv8m Benchmark Results - Throughput](https://github.com/openvinotoolkit/openvino_notebooks/assets/109281183/2d59819e-61b7-4995-bdf3-a6d1090afdd4)
**Figure 7. YOLOv8m Benchmark Results - Throughput.**
![YOLOv8m Benchmark Results - Latency](https://github.com/openvinotoolkit/openvino_notebooks/assets/109281183/bed6fc01-f0d4-4f8e-af6a-703182947232)
**Figure 8. YOLOv8m Benchmark Results - Latency.**

Benchmarking was performed on an Intel® Xeon® Platinum 8480+ (1 socket, 56 cores) running Ubuntu 22.04.2 LTS. The tests utilized the YOLOv8m model with OpenVINO 2023.0 (See Figures 7 and 8). For complete configuration, please check the Appendix section.

### Running the Benchmark

Use the following command to run the benchmark:

```shell
!benchmark_app -m $int8_model_det_path -d $device -hint latency -t 30
```

Replace `int8_model_det_path` with the path to your INT8 model and $device with the specific device you're using (CPU, GPU, etc.). This command performs inference on the model for 30 seconds. Run `benchmark_app --help` for additional command-line options.

Congratulations! You have successfully set up and run the Detection and Tracking for Automated Self-Checkout application with OpenVINO.

### Appendix

Platform Configurations for Performance Benchmarks for YOLOv8m Model

| Type Device | | CPU | | | GPU | |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| System Board | Intel Corporation<br>D50DNP1SBB | AAEON<br>UPN-ADLN01 V1.0<br>220950173 | Intel® Client Systems<br>NUC12SNKi72 | Intel Corporation<br>M50CYP2SBSTD | Intel® Client Systems<br>NUC12SNKi72 | Intel® Client Systems<br>NUC12SNKi72 |
| CPU | Intel(R) Xeon(R) <br>Platinum 8480+ | Intel® Core™ <br>i3-N305 @ 3.80 GHz | 12th Gen Intel® Core™ <br>i7-12700H @ 2.30 GHz | Intel(R) Xeon(R) <br>Gold 6348 CPU @ 2.60GHz | 12th Gen Intel® Core™ <br>i7-12700H @ 2.30 GHz | 12th Gen Intel® Core™ <br>i7-12700H @ 2.30 GHz |
| Sockets / Physical cores | 1 /  56 <br>(112 Threads) | 1 / 8 <br>(8 Threads) | 1 /14 <br>(20 Threads) | 2 / 28 <br>(56 Threads) | 1 /14 <br>(20 Threads) | 1 /14 <br>(20 Threads) |
| HyperThreading / Turbo Setting | Enabled / On | Disabled | Enabled / On | Enabled / On | Enabled / On | Enabled / On |
| Memory | 512 GB DDR4 <br>@ 4800 MHz | 16GB DDR5 <br>@4800 MHz | 64 GB DDR4 <br>@ 3200 MHz | 256 GB DDR4 <br>@ 3200 MHz | 64 GB DDR4 <br>@ 3200 MHz | 64 GB DDR4 <br>@ 3200 MHz |
| OS | Ubuntu 22.04.2 LTS | Ubuntu 22.04.2 LTS | Windows 11 <br>Enterprise v22H2 | Ubuntu 22.04.2 LTS | Windows 11 <br>Enterprise v22H2 | Windows 11 <br>Enterprise v22H2 |
| Kernel | 5.15.0-72-generic | 5.15.0-1028-intel-iotg | 22621.1702 | 5.15.0-57-generic | 22621.1702 | 22621.1702 |
| Software | OpenVINO 2023.0 | OpenVINO 2023.0 | OpenVINO 2023.0 | OpenVINO 2023.0 | OpenVINO 2023.0 | OpenVINO 2023.0 |
| BIOS | Intel Corp. <br>SE5C7411.86B.9525<br>.D13.2302071333 | American Megatrends <br>International, <br>LLC. UNADAM10 | Intel Corp. <br>SNADL357.0053<br>.2022.1102.1218 | Intel Corp. <br>SE5C620.86B.01<br>.01.0007.2210270543 | Intel Corp. <br>SNADL357.0053<br>.2022.1102.1218 | Intel Corp. <br>SNADL357.0053<br>.2022.1102.1218 |
| BIOS Release Date | 02/07/2023 | 12/15/2022 | 11/02/2022 | 10/27/2022 | 11/02/2022 | 11/02/2022 |
| GPU | N/A | N/A | 1x Intel® Arc A770™ <br>16GB, 512 EU | 1x Intel® Iris® <br>Xe Graphics | 1x Intel® Data Center <br>GPU Flex 170 | 1x Intel® Arc A770™ <br>16GB, 512 EU | 1x Intel® Iris® <br>Xe Graphics |
| Workload: <br>Codec, <br>resolution, <br>frame rate<br> Model, size (HxW), BS | Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 | Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 | Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 | Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 | Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 |  Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 | Yolov8m Model<br>– input size [640, 640], batch 1<br> FP16 \| int8 |
| TDP | 350W | 15W | 45W | 235W | 45W | 45W |
| Benchmark Date | May 31, 2023 | May 29, 2023 | June 15, 2023 | May 29, 2023 | June 15, 2023 | May 29, 2023 
| Benchmarked by | Intel Corporation | Intel Corporation | Intel Corporation | Intel Corporation | Intel Corporation | Intel Corporation |

# Additional Resources

- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca?project=ai_ref_kits/automated_self_checkout?file=README.md" />
