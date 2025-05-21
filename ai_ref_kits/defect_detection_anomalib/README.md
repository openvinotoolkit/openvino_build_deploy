<div id="top" align="center">
  <h1>Defect Detection with Anomalib and OpenVINO™ Toolkit</h1>
  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
    <a href="https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/defect_detection_anomalib/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb">📔&nbsp;Jupter Notebook: Training</a>,
    <a href="https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/defect_detection_anomalib/501b_inference_with_a_robotic_arm.ipynb">Inference&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=ho6R69EDyao">📺&nbsp;Overview&nbsp;Video&nbsp;·</a>
    <a href="https://www.youtube.com/watch?v=OifcJbZRaGM">👨‍💻&nbsp;Code&nbsp;Demo&nbsp;Video&nbsp;·</a>
    <a href="https://www.intel.com/content/www/us/en/developer/articles/training/defect-detection-with-anomalib.html">📚&nbsp;Step&#8209;by&#8209;step&nbsp;Tutorial</a>
  </h4>
</div>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

Intel’s OpenVINO™ Defect Detection with Anomalib offers a comprehensive solution to quality control through near-real-time detection of manufacturing defects. The kit uses Anomalib, an open-source deep-learning library, to simplify training, testing, and deploying anomaly detection models on public and custom datasets. Models can be exported to the OpenVINO™ Intermediate Representation and deployed on Intel hardware. Optimized for inference performance, these models are trainable on CPU, require low memory, and are well-suited for edge deployment. 

This kit uses the following technology stack:
- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) ([docs](https://docs.openvino.ai/))
- [Anomalib](https://github.com/openvinotoolkit/anomalib)

Check out our [AI Reference Kits repository](/) for other kits.

![defect-detection](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/cf933593-31f7-44a5-9cd1-fc68e8a719a9)

### What's New

New updates will be added here.

<details open><summary><b>Table of Contents</b></summary>
  
- [Installing Anomalib](#installing-anomalib)
  - [Getting Started with the Jupyter Notebook](#getting-started-with-the-jupyter-notebook)
    - [Setting up your camera](#setting-up-your-camera)
    - [Setting up the Dobot Robot (Optional)](#setting-up-the-dobot-robot-optional)
    - [Data Acquisition and Inferencing](#data-acquisition-and-inferencing)
    - [Training](#training)
- [Additional Resources](#additional-resources)

</details>

## Star the Repository

Star the [repository](https://github.com/openvinotoolkit/openvino_build_deploy) (optional, but recommended :))

# Installing Anomalib

If you have not installed all required dependencies, just run `pip install anomalib` in the same OpenVINO Notebooks environment.

## Getting Started with the Jupyter Notebook

This notebook demonstrates how NNCF compresses a model trained with Anomalib. The notebook is divided into the following sections:

- Train an Anomalib model without compression
- Train a model with NNCF compression
- Compare the performance of the two models (FP32 vs INT8)

### Setting up your Camera

Connect your USB Camera and verify it works using a simple camera application. Once it is verified, close the application.

### Setting up the Dobot Robot (Optional)

1. Install Dobot requirements (See Dobot documentation here: https://en.dobot.cn/products/education/magician.html).
2. Check all connections to the Dobot and verify it is working using the Dobot Studio.
3. Install the vent accessory on the Dobot and verify it works using Dobot Studio.
4. In the Dobot Studio, hit the "Home" button, and locate the:

![image](https://user-images.githubusercontent.com/10940214/219142393-c589f275-e01a-44bb-b499-65ebeb83a3dd.png)

a. Calibration coordinates: Initial position upper-left corner of cubes array.

b. Place coordinates: Position where the arm should leave the cubic over the conveyor belt.

c. Anomaly coordinates: Where you want to release the abnormal cube.

d. Then, replace those coordinates in the notebook

### Data Acquisition and Inferencing

For data acquisition and inferencing we will use [501b notebook](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/defect_detection_anomalib/501b_inference_with_a_robotic_arm.ipynb). There we need to identify the `acquisition` flag, **True** for _acquisition mode_ and **False** for _inferencing mode_. In acquisition mode be aware of the _normal_ or _abnormal_ folder we want to create, in this mode the notebook will save every image in the anomalib/datasets/cubes/{FOLDER} for further training. In inferencing mode the notebook won't save images, it will run the inference and show the results.

_Note_: If you don't have the robot you could jump to another notebook [501a](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/defect_detection_anomalib/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb) and download the dataset from this [link](https://github.com/openvinotoolkit/anomalib/releases/tag/dobot)

### Training

For training, we will use the [501a notebook](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/ai_ref_kits/defect_detection_anomalib/501a_training_a_model_with_cubes_from_a_robotic_arm.ipynb). In this example we are using "Padim" model and we are using Anomalib API for setting up the dataset, model, metrics, and the optimization process with OpenVINO.

# Additional Resources
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2023.0/home.html)

<p align="right"><a href="#top">Back to top ⬆️</a></p>

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=ai_ref_kits/defect_detection_anomalib&file=README.md" />
