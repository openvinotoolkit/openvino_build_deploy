# Agentic Tourism - Multi-Agent Travel Assistant

<div align="center">

# Agentic Tourism - Multi-Agent Travel Assistant with OpenVINO™ Toolkit  
A sophisticated multi-agent system that provides intelligent travel assistance using specialized AI agents for hotel search, flight booking, and video-based travel recommendations. Built with OpenVINO, MCP (Model Context Protocol), and the BeeAI framework.

  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
  </h4>
</div>

[![Apache License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

---

The Multimodal AI Visual Generator is a multimodal generative AI reference kit that demonstrates how large language models (LLMs) and diffusion-based image generation models can work together in a creative pipeline. It allows users to transform a single text prompt into detailed illustrated stories or stylized T-shirt design concepts, using optimized models for local deployment.

By combining LLM-driven prompt generation with image synthesis, the application shows how OpenVINO™ can accelerate multimodal generative AI workflows across Intel® NPUs, CPUs, integrated GPUs, and discrete GPUs. Multimodal AI Visual Generator delivers a complete pipeline, covering prompt input, scene generation, visual rendering, and PDF export.

This kit serves as a practical foundation for building real-world applications in storytelling, branding, education, and other creative domains powered by generative AI.

This kit uses the following technology stack:

- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [OpenVINO™ GenAI](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html)
- [Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html)
- [OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_what_is_openvino_model_server.html)
- [BEE AI Framework](https://github.com/i-am-bee/beeai-framework)
- [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)
- [A2A](https://github.com/a2aproject/A2A)

- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (LLM)
- [FLUX.1](https://github.com/black-forest-labs/flux) (text-to-image)
- [Streamlit](https://docs.streamlit.io/) (frontend)
- [FastAPI](https://fastapi.tiangolo.com/) (backend)

Check out our [AI Reference Kits repository](https://github.com/openvinotoolkit/openvino_build_deploy) for other kits.


```
┌─────────────────────────────────────────────────────────────┐
│                    Travel Router Agent                      │
│              (Main Coordinator & Orchestrator)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Hotel Finder │ │Flight Finder│ │Video Search │
│   Agent     │ │   Agent     │ │   Agent     │
│  (Port 9999)│ │ (Port 9998) │ │ (Port 9997) │
└─────────────┘ └─────────────┘ └─────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────┐ ┌─────────────┐
│ Travel MCP Tool Server  │ │ OpenVINO    │
│  • search_hotel         │ │ Multimodal  │
│  • search_flight        │ │ VLM + Vector│
│  • weather              │ │ Store       │
│  • think                │ └─────────────┘
└─────────────────────────┘
```

---

# To run it

## 1. Create virtual environment

```
python -m venv agentic
source agentic/bin/activate
```

## 2. Download optimized models for OVMS

```
```docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --task text_generation --tool_parser phi4
curl -L -o models/OpenVINO/Phi-4-mini-instruct-int4-ov/chat_template.jinja https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/3/extras/chat_template_examples/chat_template_phi4_mini.jinja
```

## 3. Start LLM (OVMS)
```
docker run -d --user $(id -u):$(id -g) --rm -p 8001:8001 -v $(pwd)/models:/models openvino/model_server:latest \
--rest_port 8001 --model_repository_path models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --tool_parser phi4 --cache_size 2 --task text_generation --enable_prefix_caching true
```

## 4. Get your Amadeus Key and replace it on config/mcp_tools.yaml (client_id and client_secret)

Go to  from https://developers.amadeus.com/self-service/apis-docs/guides/developer-guides/quick-start/


## 5. Start Tools (MCP)

Terminal 1  
```
python mcp_tools/tourism_search.py 

```
## 6. Start Agents

Terminal 2

```
python agents/agent_runner.py --agent hotel_finder
```

Terminal 3  
```
python agents/agent_runner.py --agent flight_finder

```
## 7. Start UI
Terminal 4  
```
python start_ui.py
```