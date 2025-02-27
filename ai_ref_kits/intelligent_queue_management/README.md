<div id="top" align="center">
  <h1>Intelligent Queue Management with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="docs/convert-and-optimize-the-model.ipynb">📔&nbsp;Jupyter Notebook: Convert</a>,
    <a href="docs/run-the-application.ipynb">Run&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=fwFbl4_8jk8">📺&nbsp;Overview&nbsp;Video&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=9E2baweCCXQ">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video&nbsp;·</a>
    <a href="https://www.intel.com/content/www/us/en/developer/articles/training/create-intelligent-queue-management.html">📚&nbsp;Step&#8209;by&#8209;step&nbsp;Tutorial</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

The Intelligent Queue Management kit is designed to help businesses manage customer queues more effectively by analyzing camera video streams and detecting the number of people in each queue. By leveraging this data, the system optimizes the queuing process to reduce customer wait times. The kit ensures efficient performance and seamless integration by utilizing OpenVINO™, a toolkit that enables developers to deploy deep learning models across various hardware platforms.

This kit uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [Ultralytic YOLOv8](https://github.com/ultralytics/ultralytics)

Check out our [AI Reference Kits repository](/) for other kits.

![intelligent-queue-management](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/6874cfe8-3462-4b30-8026-c14aab7b695c)

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Converting and Optimizing the Model](#converting-and-optimizing-the-model)
  - [Running the Application](#running-the-application)
- [Benchmarking the Model with OpenVINO's `Benchmark_App`](#benchmarking-the-model-with-openvinos-benchmark_app)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project.

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Installing Prerequisites

This project requires Python 3.9 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

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
cd openvino_build_deploy/ai_ref_kits/intelligent_queue_management
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

#### Activating the Environment

Activate the virtual environment using the following command:

```shell
source venv/bin/activate   # For Unix-based operating systems such as Linux or macOS
```

_NOTE: If you are using Windows, use `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

#### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

## Converting and Optimizing the Model

To convert and optimize the YOLOv8 model, run the following command:

```shell
python convert_and_optimize.py --model_name yolov8m --model_dir model --data_dir data --quantize True
```
This will convert the YOLOv8 model to an OpenVINO™ Intermediate Representation (IR) format and optimize it for use with OpenVINO™.
You can run the Python script or check out [convert-and-optimize-the-model.ipynb](docs/convert-and-optimize-the-model.ipynb) to learn more.

## Running the Application

To run the application, use the following command:

```shell
python app.py --stream data/sample_video.mp4 --model_path model/yolov8m_openvino_int8_model/yolov8m.xml --zones_config_file config/zones.json --customers_limit 3
```
This will run the application with the specified arguments. Replace "video_file.mp4" with the path to your input video file, "zones.json" with the path to your zones configuration file, and "3" with the maximum number of customers allowed in the queue.
You can also run the [run-the-application.ipynb](docs/run-the-application.ipynb) to learn more about the inference process. To stop the application at any time, press 'q' or escape.

_NOTE: Alternatively, you can run all steps with the following command:_

```shell
python main.py --stream data/sample_video.mp4
```

<p align="right"><a href="#top">Back to top ⬆️</a></p>

# Benchmarking the Model with OpenVINO's `Benchmark_App`

Benchmarking provides insight into your model's real-world performance. Performance may vary based on use and configuration.

### Benchmark Results 

![YOLOv8m Benchmark Results](https://github.com/openvinotoolkit/openvino_notebooks/assets/109281183/2d59819e-61b7-4995-bdf3-a6d1090afdd4)
![](https://github.com/openvinotoolkit/openvino_notebooks/assets/109281183/bed6fc01-f0d4-4f8e-af6a-703182947232)

Benchmarking was performed on an Intel® Xeon® Platinum 8480+ (1 socket, 56 cores) running Ubuntu 22.04.2 LTS. The tests utilized the YOLOv8m model with OpenVINO 2023.0. For complete configuration, please check the Appendix section.

### Running the Benchmark

Use the following command to run the benchmark:

```shell
!benchmark_app -m $int8_model_det_path -d $device -hint latency -t 30
```
Replace `int8_model_det_path` with the path to your INT8 model and $device with the specific device you're using (CPU, GPU, etc.). This command performs inference on the model for 30 seconds. Run `benchmark_app --help` for additional command-line options.

Congratulations! You have successfully set up and run the Intelligent Queue Management application with OpenVINO™.

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
| OpenVINO Version | 2023.0.0-10926 | 2023.0.0-10926 | 2023.0.0-10926 | 2023.0.0-10926 | 2023.0.0-10926 | 2023.0.0-10926 |
| Benchmark Date | May 31, 2023 | May 29, 2023 | June 15, 2023 | May 29, 2023 | June 15, 2023 | May 29, 2023 
| Benchmarked by | Intel Corporation | Intel Corporation | Intel Corporation | Intel Corporation | Intel Corporation | Intel Corporation |

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>
