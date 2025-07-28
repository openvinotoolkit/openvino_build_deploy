<div id="top" align="center">
  <h1>Explainable AI with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="explainable_ai.ipynb">📔&nbsp;Jupyter&nbsp;Notebook&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=InXb2wLCsJE">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

Explainable AI uses data quality measurements and saliency maps to understand the predictions and performance of computer vision models during inference. Data and model explainability provide insights into how predictions are made, helping refine models for efficiency and performance. This application utilizes the Intel OpenVINO™ toolkit, enabling seamless deployment of deep learning models across hardware platforms.

This kit uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [OpenVINO Datumaro](https://docs.openvino.ai/2023.3/datumaro_documentation.html)
- [Ultralytic YOLOv8](https://github.com/ultralytics/ultralytics)

You can also explore [OpenVINO™ Model Hub](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html) to view performance benchmarks for models like YOLO.

Check out our [AI Reference Kits repository](/) for other kits.

![explainable-ai](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/14958589-433b-4617-b3ea-e2307fe6cb53)

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting Up Your Environment](#setting-up-your-environment)
  - [Running the Application](#running-the-application)
- [Benchmarking the Model with OpenVINO's `Benchmark_App`](#benchmarking-the-model-with-openvinos-benchmark_app)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

## Installing Prerequisites

Now, let's dive into the steps starting with installing Python. We recommend using Ubuntu to set up and run this project. This project requires Python 3.10 or higher and a few libraries. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

Install libraries and tools:

```shell
sudo apt install git git-lfs gcc python3-venv python3-dev
```

_NOTE: If you are using Windows, you would need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) also._

## Setting Up Your Environment

### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

The above will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/ai_ref_kits/explainable_ai
```

Then pull the video sample:

```shell
git lfs -X= -I="Cars-FHD.mov" pull
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

_NOTE: If you are using Windows, use the `venv\Scripts\activate` command instead._

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

_NOTE: Datumaro contains C++ and Rust implementations to improve Python performance: Please ensure you install the [Rust toolchain](https://www.rust-lang.org/tools/install) in your system to run this sample.

## Running the Application

You can run [explainable_ai.ipynb](explainable_ai.ipynb) to learn more about the inference process. This notebook contains detailed instructions to run the Explainable AI application, load and analyze a short data section with data quality metrics, and generate saliency maps with an OpenVINO YOLOv8 model using Ultralytics. For the data quality metrics generation, we leverage the open-source toolkit datumaro, including specifically the tutorial [here](https://github.com/openvinotoolkit/datumaro/blob/develop/notebooks/11_validate.ipynb). This edge AI reference kit focuses on a specific digital transportation use case, with an analysis of only a few data quality metrics—please visit the [Datumaro tutorials](https://github.com/openvinotoolkit/datumaro/tree/develop/notebooks) for resources on how to perform advanced data exploration, and explore and remediate more types of data quality issues.

Congratulations! You have successfully set up and run the Explainable AI kit.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

# Benchmarking the Model with OpenVINO's `benchmark_app`

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

You can evaluate performance benchmarks for the YOLO model across a range of Intel® platforms using the [OpenVINO™ Model Hub](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/model-hub.html). The Model Hub is a curated resource for developers to explore OpenVINO-optimized models and assess their latency and throughput on Intel® CPUs, integrated GPUs, discrete GPUs, and NPUs.

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
- [DarwinAI Case Study](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/partners/documents/darwinai-delivers-explainable-ai-case-study.html): See how others are implementing Explainable AI practices with Intel.
- [Interview on Building Ethical AI with Explainable AI](https://www.youtube.com/watch?v=wWjlWpI4EIE): Learn more about key topics around Explainable AI from Ria, our evangelist ​and creator of the Explainable AI kit.

<p align="right"><a href="#top">Back to top ⬆️</a></p>

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=ai_ref_kits/explainable_ai&file=README.md" />
