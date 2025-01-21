<div id="top" align="center">
  <h1>Smart Meter Scanning with OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="https://github.com/openvinotoolkit/openvino_notebooks/blob/recipes/notebooks/203-meter-reader/203-meter-reader.ipynb">📔&nbsp;Jupyter&nbsp;Notebook&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=y2xCZYe8GAQ">📺&nbsp;Overview&nbsp;Video&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=9jcFGzFjHXo">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video&nbsp;·</a>
    <a href="https://www.intel.com/content/www/us/en/developer/articles/training/create-smart-meter-scanning.html">📚&nbsp;Step&#8209;by&#8209;step&nbsp;Tutorial</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

Smart Meter Scanning is an AI-based application that enables cameras to automatically read results from analog meters, transforming those results into digital data with accurate, near-real-time meter results. It uses computer vision, object detection, and object segmentation.

This kit uses the following technology stack:
- [OpenVINO toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([Docs](https://docs.openvino.ai/))
- [Models from PaddlePaddle](https://github.com/PaddlePaddle)

Check out our [AI Reference Kits repository](/) for other kits.

![smart-meter-scanning](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/0136d123-15c9-4696-bf4d-b169b3c7db4d)

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Getting Started](#getting-started)
  - [Installing Prerequisites](#installing-prerequisites)
  - [Setting up your Environment](#setting-up-your-environment)
  - [Running the Application](#running-the-application)
- [Additional Resources](#additional-resources)

</details>

# Getting Started

Now, let's dive into the steps, starting with installing Python. 

## Installing Prerequisites

This project requires Python 3.9 or higher. If you don't have Python installed on your machine, go to https://www.python.org/downloads/ and download the latest version for your operating system. Follow the prompts to install Python, making sure to check the option to add Python to your PATH environment variable.

## Setting up your Environment

### Cloning the Repository

To clone the repository, run the following command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

This will clone the repository into a directory named "openvino_build_deploy" in the current directory. Then, navigate into the directory using the following command:

```shell
cd openvino_build_deploy/ai_ref_kits/meter_reader
```

Then pull the models:

```shell
git lfs -X= -I=model/ pull
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

NOTE: If you are using Windows, use `venv\Scripts\activate` command instead.

This will activate the virtual environment and change your shell's prompt to indicate that you are now working within that environment.

### Installing the Packages

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

### Preparing your Models 

Prepare your detection and segmentation models with this command: 
```shell
cd model
sudo sh ./download_pdmodel.sh
```

## Running the Application

To run the application, use the following command:

```shell
python main.py -i data/test.jpg -c config/config.json  -t "analog"
```

This will run the application with the specified arguments. Replace "data/test.jpg" with the path to your input image.
The result images will be exported to the same test image folder. You can also run the [meter-reader.ipynb](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/meter-reader) to learn more about the inference process.

In case you have trouble downloading the detection and segmentation models from PaddlePaddle, we have also provided YOLOV8 and deeplabv3 as the detection and segmentation models in the "model" folder. You could then run the application using these two models by swtiching the configuration file to "config/yolov8.json" with the following command:

```shell
python main.py -i data/test.jpg -c config/yolov8.json  -t "analog"
```

Congratulations! You have successfully set up and run the Automatic Industrial Meter Reading application with OpenVINO™.

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>
