# Edge AI Reference Kits for OpenVINO™ Toolkit

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

[Edge AI Reference Kits](https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html) are fully functioning applications demonstrating deep learning AI use cases. You can leverage these pre-built components and code samples as the basis for solutions in industries like retail, healthcare, manufacturing, and more. Start development of your AI application on top of our kits, or integrate into your existing apps. 

## Table of Contents

- [Available Kits](#available-kits)
	- [🚶 Intelligent Queue Management](#-intelligent-queue-management)
	- [🔍 Defect Detection](#-defect-detection)
	- [⏲️ Smart Meter Scanning](#%EF%B8%8F-smart-meter-scanning)
	- [🛍️ Automated Self-Checkout](#%EF%B8%8F-automated-self-checkout)
	- [🗣️ Custom AI Assistant](#%EF%B8%8F-custom-ai-assistant)
	- [🔦 Explainable AI](#-explainable-ai)
   	- [🎨 Multimodal AI Visual Generator](#-multimodal-ai-visual-generator)
	- [💬 Conversational AI Chatbot](#-conversational-ai-chatbot)
	- [🛒 AI Insight Agent with RAG](#-AI-Insight-Agent-with-RAG)

- [Troubleshooting and Resources](#troubleshooting-and-resources)

## Available Kits
Currently, there are the following kits:

### 🚶 Intelligent Queue Management
[![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/6874cfe8-3462-4b30-8026-c14aab7b695c)](intelligent_queue_management)

| [Intelligent Queue Management](intelligent_queue_management) |  |
| - | - |
| Related AI concepts | object detection, object counting, YOLOv8 |
| Example industries | retail business, healthcare |
| Overview | [Solution video](https://www.youtube.com/watch?v=fwFbl4_8jk8) |
| Demo | [Code video](https://www.youtube.com/watch?v=9E2baweCCXQ) |
| Tutorial | [Step-by-step article](https://www.intel.com/content/www/us/en/developer/articles/training/create-intelligent-queue-management.html) |

The solution for managing customer queues more effectively by analyzing camera video streams and detecting the number of people in each queue. The system then uses this real time data to optimize the queuing process and reduce customer waiting times.

### 🔍 Defect Detection
[![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/cf933593-31f7-44a5-9cd1-fc68e8a719a9)](defect_detection_anomalib)

| [Defect Detection with Anomalib](defect_detection_anomalib) |  |
| - | - |
| Related AI concepts | object detection, anomaly detection, unsupervised learning, Padim |
| Example industries | manufacturing, healthcare, agriculture |
| Overview | [Solution video](https://www.youtube.com/watch?v=ho6R69EDyao) |
| Demo | [Code video](https://www.youtube.com/watch?v=OifcJbZRaGM) |
| Tutorial | [Step-by-step article](https://www.intel.com/content/www/us/en/developer/articles/training/defect-detection-with-anomalib.html) |

The solution offers a comprehensive solution to quality control. It provides companies and their technical teams a single-source, end-to-end solution to catch manufacturing defects in real time.

### ⏲️ Smart Meter Scanning
[![image](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/0136d123-15c9-4696-bf4d-b169b3c7db4d)](meter_reader)

| [Smart Meter Scanning](meter_reader) |  |
| - | - |
| Related AI concepts | object detection, object segmentation, OCR |
| Example industries | energy, manufacturing |
| Overview | [Solution video](https://www.youtube.com/watch?v=y2xCZYe8GAQ) |
| Demo | [Code video](https://www.youtube.com/watch?v=9jcFGzFjHXo) |
| Tutorial | [Step-by-step article](https://www.intel.com/content/www/us/en/developer/articles/training/create-smart-meter-scanning.html) |

This solution implements automated reading of industrial meters, using cameras and computer vision. By transforming analog data into digital, companies have access to an automated intake of real time data that allows them to make informed decisions and become more efficient with resource usage.

### 🛍️ Automated Self-Checkout
![automated-self-checkout](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/965a6604-fa15-427e-9d44-c23fa0bbeb6b)

| [Automated Self-Checkout](automated_self_checkout) |  |
| - | - |
| Related AI concepts | object detection |
| Example industries | retail |
| Overview | [Solution video](https://www.youtube.com/watch?v=VrJRr_thxcs) |
| Demo | [Code video](https://www.youtube.com/watch?v=rw8cTr-hD-g) |
| Tutorial | [Step-by-step article](https://www.intel.com/content/www/us/en/developer/articles/training/detect-objects-openvino-automated-self-checkout.html) |

This solution is designed to help automate checkout for retail businesses more effectively, by analyzing video streams and detecting and tracking interactions with retail products.

### 🗣️ Custom AI Assistant
[![custom-ai-assistant](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/e0c2f2db-c213-4071-970b-09ebc1eea710)](custom_ai_assistant)

| [Custom AI Assistant](custom_ai_assistant) |  |
|----------------------------------------------------| - |
| Related AI concepts                                | Speech Recognition, Natural Language Understanding, Large Language Models (LLMs), Generative AI |
| Example industries                                 | Retail Customer Service, Smart Home Automation |
| Demo                                               | [Code video](https://www.youtube.com/watch?v=9jnY9PJjYVk) |

The Custom AI Assistant, powered by the OpenVINO™ toolkit, integrates voice-activated interfaces into various applications. It employs advanced models for speech recognition, natural language processing, and speech synthesis. Additionally, the agent leverages the capabilities of Large Language Models (LLMs) and Generative AI, enabling it to understand complex queries, generate coherent responses, and offer a more dynamic and interactive communication experience. These advanced AI technologies enhance the agent's ability to engage in natural, human-like conversations, making it ideal for applications in customer service and smart home automation.

### 🔦 Explainable AI
![explainable-ai](https://github.com/openvinotoolkit/openvino_notebooks/assets/138901786/14958589-433b-4617-b3ea-e2307fe6cb53)

| [Explainable AI](explainable_ai) |  |
| - | - |
| Related AI concepts | object detection |
| Example industries | digital transporation |
| Demo | [Code video](https://www.youtube.com/watch?v=InXb2wLCsJE) |

Understanding why computer vision models make certain predictions using data and model explainability can help us refine our models to be more efficient and performant. This solution demonstrates how to leverage the OpenVINO™ toolkit, Datumaro, and Ultralytics to generate data quality measurements and saliency maps to understand the predictions and performance of computer vision models during inference.

### 🎨 Multimodal AI Visual Generator
![multimodal-ai-visual-generator](https://github.com/user-attachments/assets/43d6e473-19c1-4047-aee1-07a484cd0dc1)

| [Multimodal AI Visual Generator](multimodal_ai_visual_generator) |  |
| - | - |
| Related AI concepts | Large Language Models (LLMs), Image Generation, Multimodal AI |
| Example industries |  Retail and E-commerce, Creative Design, Digital Marketing |
| Demo |  |

Multimodal AI Visual Generator is a generative AI reference kit that transforms a single creative prompt into either a multi-scene illustrated story or a set of stylized T-shirt design ideas. The pipeline combines an instruction-tuned LLM (Qwen2) and a diffusion-based image generator (FLUX.1), both optimized with OpenVINO™. The kit features a FastAPI backend and Streamlit UI for a responsive, end-to-end user experience, and supports PDF export for sharing outputs.

### 💬 Conversational AI Chatbot
[![conversational-ai-chatbot](https://github.com/user-attachments/assets/262ba878-b845-445b-aff7-0a118517d409)](conversational-ai-chatbot)

| [Conversational AI Chatbot](conversational_ai_chatbot) |                                                                                                                                                         |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Related AI concepts                                    | Speech Recognition, Natural Language Understanding, Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Speech Synthesis, Generative AI |
| Example industries                                     | Tourism                                                                                                                                                 |
| Demo                                                   |                                                                                                                                                         |

The Conversational AI Chatbot is an open-source, voice-driven chat agent that answers spoken questions with meaningful, spoken responses.  It can be configured to respond in any type of scenario or context. 
This kit demonstrates the AI Chatbot’s capabilities by simulating the experience of talking to a hotel concierge.

### 🛒 AI Insight Agent with RAG
[![agentic_llm_rag](https://github.com/user-attachments/assets/0471ab91-ded5-4a5f-8d8e-5432f1b4b45c)](agentic-llm-rag)

| [AI Insight Agent with RAG](agentic_llm_rag) |                                                                                                                                                         |
|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Related AI concepts                                    | Natural Language Understanding, Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Agentic AI, Generative AI |
| Example industries                                     | Retail                                                                                                                                                 |
| Demo                                                   |                                                                                                                                                         |

The AI Insight Agent with RAG uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to interpret user prompts, engage in meaningful dialogue, perform calculations, use RAG techniques to improve its knowledge and interact with the user to add items to a virtual shopping cart.

## Troubleshooting and Resources
- Open a [discussion topic](https://github.com/openvinotoolkit/openvino_build_deploy/discussions)
- Create an [issue](https://github.com/openvinotoolkit/openvino_build_deploy/issues)
- Learn more about [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- Explore [OpenVINO’s documentation](https://docs.openvino.ai/2024/home.html)

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=ai_ref_kits&file=README.md" />
