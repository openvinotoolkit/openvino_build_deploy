# Interactive Demos for OpenVINO™ Toolkit

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

This directory contains interactive demos aimed at demonstrating how OpenVINO performs as an AI optimization and inference engine (cloud, client, and edge), and attracting more people to Intel's booth at events. 

_Please note: even though you can find here a lot of useful OpenVINO code, education is not the main purpose of this directory. If you are interested in educational content visit [ai_ref_kits](../ai_ref_kits/README.md)._

## Available Demos

Currently, the following demos are tested for Python :warning:3.10-3.13 :warning: only:

### 🎨 Paint Your Dreams

The demo generates images in a reasonable time (seconds) on Intel hardware.

[![image](https://github.com/user-attachments/assets/d4d298e6-1df2-44bf-ae88-ba9771efeb2c)](paint_your_dreams_demo)

| [Paint Your Dreams](paint_your_dreams_demo)   |                                                            |
|-----------------------------------------------|------------------------------------------------------------|
| Related AI concepts                           | Visual GenAI, LCM, image generation                        |
| Platforms                                     | Client (CPU, GPU), Cloud (CPU, GPU)                        |
| Owner                                         | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🏥 Virtual Assistant

The demo works as a virtual AI assistant. The default personality is a healthcare assistant, whose task is to greet the patient and summarize the patient's condition based on the conversation and uploaded examination report.

[![image](https://github.com/user-attachments/assets/f1ca6a23-0a5d-4a7d-94d4-89d0ef2b68ea)](virtual_ai_assistant_demo)

| [Virtual AI Assistant](virtual_ai_assistant_demo) |                                                            |
|---------------------------------------------------|------------------------------------------------------------|
| Related AI concepts                               | LLM, RAG, GenAI, Llama3, LLamaIndex                        |
| Platforms                                         | Client (CPU, GPU, NPU), Cloud (CPU, GPU)                   |
| Owner                                             | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🚶 People Counter

The demo counts people (or any other selected object) standing in front of the webcam, presenting differences in performance between various precisions and devices on the used platform.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/e386c632-34f3-41c7-9713-c5aca8c1842a)](people_counter_demo)

| [People Counter](people_counter_demo) |                                                           |
|---------------------------------------|-----------------------------------------------------------|
| Related AI concepts                   | object detection, object counting, YOLO11, YOLOv8         |
| Platforms                             | Client (CPU, GPU, NPU), Edge (CPU)                        |
| Owner                                 | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🏭 Spot the Object

The demo detects, tracks and counts defined objects in front of the webcam. The default object is a hazelnut, but it can be changed to any other object. It works especially good with a conveyor belt.

[![image](https://github.com/user-attachments/assets/103617f8-e895-4cc0-9ed9-60a1e87b8706)](spot_the_object_demo)

| [Spot the Object](spot_the_object_demo) |                                                               |
|-----------------------------------------|---------------------------------------------------------------|
| Related AI concepts                     | object detection, object tracking, object counting, YOLOWorld |
| Platforms                               | Client (CPU, GPU, NPU), Edge (CPU)                            |
| Owner                                   | [@adrianboguszewski](https://github.com/adrianboguszewski)    |

### 🔮 Hide Your Mess Behind

The demo blurs the background behind a person on a webcam. The app is built using NodeJS and Electron technologies. It can be run from the compiled exe file or using npm.

[![image](https://github.com/user-attachments/assets/e6925e6b-0d81-41da-b9b0-c4f21f173681)](hide_your_mess_behind_demo)

| [Hide Your Mess Behind](hide_your_mess_behind_demo) |                                                                                       |
|--------------------------------------|------------------------------------------------------------------------------------------------------|
| Related AI concepts                  | image segmentation                                                                                   |
| Platforms                            | Client (CPU, GPU, NPU), Edge (CPU)                                                                   |
| Owner                                | [@Roszczyk](https://github.com/Roszczyk), [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 💃 Strike a pose

The demo estimates poses of all people standing in front of the webcam.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/3bff0def-9050-450f-8699-389defec4136)](strike_a_pose_demo)

| [Strike a Pose](strike_a_pose_demo) |                                                            |
|-------------------------------------|------------------------------------------------------------|
| Related AI concepts                 | pose estimation, YOLO11, YOLOv8                            |
| Platforms                           | Client (CPU, GPU, NPU), Edge (CPU)                         |
| Owner                               | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🎅 🎃 🐰 Theme Demo

This is a funny demo making people look like something/someone else based on the chosen theme. It brings much attention to the booth. The available themes are: Christmas, Halloween and Easter.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/0cbf768c-0260-41bc-af64-00dfdaa9110c)](theme_demo)
[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/b289b9f0-1c5b-4cae-ae0b-ea905d05d5e5)](theme_demo)
[![image](https://github.com/user-attachments/assets/425e0ecd-2ff6-42f1-881a-d028465038fc)](theme_demo)

| [Theme Demo](theme_demo) |                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------|
| Related AI concepts      | face detection, landmarks regression, emotion recognition, pose estimation, OpenPose |
| Platforms                | Client (CPU, GPU, NPU), Edge (CPU)                                                   |
| Owner                    | [@adrianboguszewski](https://github.com/adrianboguszewski)                           |

## Contributing

Please follow the rules defined in the [contributing guide](CONTRIBUTING.md) to add a demo to this repository.

## Troubleshooting and Resources
- Open a [discussion topic](https://github.com/openvinotoolkit/openvino_build_deploy/discussions)
- Create an [issue](https://github.com/openvinotoolkit/openvino_build_deploy/issues)
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=demos&file=README.md" />
