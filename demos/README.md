# Interactive Demos for OpenVINO™ Toolkit

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

This directory contains interactive demos aimed at demonstrating how OpenVINO performs as an AI optimization and inference engine (cloud, client, and edge), and attracting more people to Intel's booth at events. 

_Please note: even though you can find here a lot of useful OpenVINO code, education is not the main purpose of this directory. If you are interested in educational content visit [ai_ref_kits](../ai_ref_kits/README.md)._

## Available Demos

Currently, there are the following demos:

### 🎨 Paint Your Dreams

The demo generates images in a reasonable time (seconds) on Intel hardware.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/a7f53cf2-a40b-4eb2-bb9a-72969ce8ad04)](paint_your_dreams_demo)

| [Paint Your Dreams](paint_your_dreams_demo)   |                                                            |
|-----------------------------------------------|------------------------------------------------------------|
| Related AI concepts                           | Visual GenAI, LCM, image generation                        |
| Platforms                                     | Client (CPU, GPU), Cloud (CPU, GPU)                        |
| Owner                                         | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🚶 People Counter

The demo counts people standing in front of the webcam, presenting differences in performance between various precisions and devices on the used platform.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/e386c632-34f3-41c7-9713-c5aca8c1842a)](people_counter_demo)

| [People Counter](people_counter_demo) |                                                            |
|---------------------------------------|------------------------------------------------------------|
| Related AI concepts                   | object detection, object counting, YOLOv8                  |
| Platforms                             | Client (CPU, GPU, NPU), Edge (CPU)                         |
| Owner                                 | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🏥 Healthcare Assistant

The demo works as a virtual healthcare assistant, whose task is to greet the patient and summarize the patient's condition based on the conversation and uploaded examination report. 

[![image](https://github.com/user-attachments/assets/28e66746-1f6c-4dfc-b345-ae9f0e003e76)](healthcare_assistant_demo)

| [Healthcare Assistant](healthcare_assistant_demo) |                                                            |
|---------------------------|------------------------------------------------------------|
| Related AI concepts       | LLM, RAG, ASR, GenAI, Llama3, Whisper                      |
| Platforms                 | Client (CPU, GPU), Cloud (CPU, GPU)                        |
| Owner                     | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 💃 Strike a pose

The demo estimates poses of all people standing in front of the webcam.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/3bff0def-9050-450f-8699-389defec4136)](strike_a_pose_demo)

| [Strike a Pose](strike_a_pose_demo) |                                                            |
|-------------------------------------|------------------------------------------------------------|
| Related AI concepts                 | pose estimation, Open Pose                                 |
| Platforms                           | Client (CPU, GPU, NPU), Edge (CPU)                         |
| Owner                               | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🎃 Spooky Demo

The demo detects people in front of the webcam and changes them into skeletons with a pumpkin head.

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/b289b9f0-1c5b-4cae-ae0b-ea905d05d5e5)](spooky_demo)

| [Spooky Demo](spooky_demo) |                                                            |
|----------------------------|------------------------------------------------------------|
| Related AI concepts        | pose estimation, Open Pose                                 |
| Platforms                  | Client (CPU, GPU, NPU), Edge (CPU)                         |
| Owner                      | [@adrianboguszewski](https://github.com/adrianboguszewski) |

### 🎅 Santa Claus Demo

The demo detects people in front of the webcam and changes them into Santa Claus (the biggest face) and reindeer (all other faces).

[![image](https://github.com/openvinotoolkit/openvino_build_deploy/assets/4547501/0cbf768c-0260-41bc-af64-00dfdaa9110c)](santa_claus_demo)

| [Santa Claus Demo](santa_claus_demo) |                                                            |
|--------------------------------------|------------------------------------------------------------|
| Related AI concepts                  | face detection, landmarks regression, emotion recognition  |
| Platforms                            | Client (CPU, GPU, NPU), Edge (CPU)                         |
| Owner                                | [@adrianboguszewski](https://github.com/adrianboguszewski) |

## Contributing

Please follow the rules defined in the [contributing guide](CONTRIBUTING.md) to add a demo to this repository.

## Troubleshooting and Resources
- Open a [discussion topic](https://github.com/openvinotoolkit/openvino_build_deploy/discussions)
- Create an [issue](https://github.com/openvinotoolkit/openvino_build_deploy/issues)
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)
