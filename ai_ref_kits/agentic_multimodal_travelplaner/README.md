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
│                        (Port 9996)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
   ┌───────────────-──────┼───────────────┬───────────────┐
   │                      │               │               │
   ▼                      ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐
│Hotel Finder │ │Flight Finder│ │Budget Agent │ │Video Processing│
│   Agent     │ │   Agent     │ │   Agent     │ │    Agent       │
│  (Port 9999)│ │ (Port 9998) │ │ (Port 9997) │ │   (Port 9995)  │
└─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘
      │               │              │               │
      ▼               ▼              ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐
│Hotel Search │ │Flight Search│ │   (Logic)   │ │Video Analysis  │
│ MCP Server  │ │ MCP Server  │ │   No MCP    │ │   MCP Server   │
│ (Port 7901) │ │ (Port 7902) │ │   Needed    │ │   (Port 7903)  │
└─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           Hardware OpenVINO AI Stack                       │
│  • Phi-4-mini-instruct (LLM) • Qwen3-8B (Video Analysis)   │
│  • Bridge Tower (Multimodal VLM) • Vector Store            │
└─────────────────────────────────────────────────────────────┘

```

---

# STEPS
- Start OVMS LLMs
- Start MCP servers
- Start Agents (start_agents.py)
- Start UI (start_ui.py)


# Setting Up Your Environment



To set up your environment, you first clone the repository, then create a virtual environment, activate the environment, and install the packages.

### Clone the Repository

To clone the repository, run this command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

This command clones the repository into a directory named "openvino_build_deploy" in the current directory. After the directory is cloned, run the following command to go to that directory:


```shell
cd openvino_build_deploy/ai_ref_kits/agentic_multimodal_travelplaner
```

`## Create virtual environment

```
python3 -m venv agentic
```

### Activate the Environment

The command you run to activate the virtual environment you created depends on whether you have a Unix-based operating system (Linux or macOS) or a Windows operating system.

To activate  the virtual environment for a **Unix-based** operating system, run:

```shell
source agentic/bin/activate   # For Unix-based operating systems such as Linux or macOS
```

To activate the virtual environment for a **Windows** operating system, run:

```shell
agentic\Scripts\activate  # This command is for Windows operating systems
```
This activates the virtual environment and changes your shell's prompt to indicate that you are now working in that environment.

### Install the Requirements

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

# Getting the LLM for agents ready with OpenVINO model Server (OVMS) using optimized models

Windows and Linux
## OPTION 1: Windows

TBC

## OPTION 2: Linux
v
### Docker Installation


### Get OVMS image

```
docker pull openvino/model_server:latest
```
### Download optimized models for OVMS

Agents LLM **Phi-4-mini-instruct-int4-ov**
```
docker run -d --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models openvino/model_server:latest \
--pull --model_repository_path /models --source_model OpenVINO/Phi-4-mini-instruct-int4-ov --tool_parser phi4 --cache_size 2 --task text_generation --enable_prefix_caching true
```

VLM **Phi-3.5-vision-instruct-int4-ov**
```
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest  \
--pull --model_repository_path /models --source_model OpenVINO/Phi-3.5-vision-instruct-int4-ov --task text_generation --pipeline_type VLM
```

Embeddings

Whisper
```
docker run --user $(id -u):$(id -g) --rm \
  -v $(pwd)/models:/models:rw \
  openvino/model_server:latest \
  --pull \
  --model_repository_path /models \
  --source_model OpenVINO/whisper-large-v3-int4-ov \
  --task text_generation \
  --pipeline_type AUTO
```

### Start OVMS

LLM
```
docker run -d --user $(id -u):$(id -g) --rm \
  -p 8000:8000 \
  -v $(pwd)/models:/models openvino/model_server:latest \
  --rest_port 8000 \
  --model_repository_path models \
  --source_model OpenVINO/Phi-4-mini-instruct-int4-ov \
  --tool_parser phi4 \
  --cache_size 2 \
  --task text_generation \
  --enable_prefix_caching true
```

VLM
```
docker run -d --rm \
  -p 8003:8000 \
  -v $(pwd)/models:/models:ro \
  openvino/model_server:latest \
  --rest_port 8000 \
  --model_name penVINO/Phi-3.5-vision-instruct-int4-ov \
  --model_path /models/OpenVINO/Phi-3.5-vision-instruct-int4-ov
```

Whisper
```
docker run -d --rm \
  -p 8004:8000 \
  -v $(pwd)/models:/models openvino/model_server:latest \
  --rest_port 8000 \
  --model_repository_path models \
  --source_model OpenVINO/whisper-large-v3-int4-ov 
```

### Check
Run
```
docker ps
```

You should see 
```
CONTAINER ID   IMAGE                          COMMAND                  CREATED       STATUS       PORTS                                         NAMES
4c1590b2d392   openvino/model_server:latest   "/ovms/bin/ovms --re…"   5 hours ago   Up 5 hours   0.0.0.0:8001->8001/tcp, [::]:8001->8001/tcp   infallible_davinci
```

We have now our LLM running and ready to be used by our agents.

# Define your Agents and tools

## MCP Servers YAML config (WIP)

# Start MCP tools
In this example, we will have multiple MCP servers to be consumed by the agents. In order to follow and understand the behavior, each agent and tool should be started from an independent terminal (be sure to activate the environment).
### Get your SerpAPi

Go to https://serpapi.com/

## Option 1 (launch all MCP servers at once) WIP

## Option 2 (Ideal for debugging) : Launch each mcp in an individual terminal

### Start Video Tool

```
cd mcp_tools
python video_tool.py
```

### Start Hotel finder API Tool (TERMINAL 1)
Clone server from AI Builder
```
cd mcp_tools
wget -O /home/ubuntu/openvino_build_deploy/workshops/Agentic_Tourism_copy/mcp_tools/ai_builder_mcp_hotel_finder.py \
https://raw.githubusercontent.com/intel/intel-ai-assistant-builder/main/mcp/mcp_servers/mcp_google_hotel/server.py

```
Start MCP tool
```
export SERP_API_KEY= 
python ai_builder_mcp_hotel_finder.py start --port 7901 --protocol sse
```

### Start Flight finder API Tool (TERMINAL 2)
Clone server from AI Builder
```
cd mcp_tools
wget -O /home/ubuntu/openvino_build_deploy/workshops/Agentic_Tourism_copy/mcp_tools/ai_builder_mcp_flight_finder.py \
https://raw.githubusercontent.com/intel/intel-ai-assistant-builder/main/mcp/mcp_servers/mcp_google_flight/server.py

```
Start MCP tool
```
export SERP_API_KEY= 
python ai_builder_mcp_flight_finder.py start --port 7902 --protocol sse
```

### Start VideoProcessing Tools (WIP)



## Agents YAML config

All agents are created by a runner `agent_runner.py` which reads two files : `config/agents_config.yaml` and `config\agents_prompt.yaml`

You can start the agents all together by running 
```python
python start_agents.py
```

### Agents configuration

### Agents Prompts





# Start Agents

### Regular A2A

Hotel finder agent (TERMINAL 3)

```
cd agents
python agent_runner_copy.py --agent travel hotel_finder
```

Flight finder agent (TERMINAL 4)
```
cd agents
python agent_runner_copy.py --agent flight_finder
```


Budget approval agent (TERMINAL 5)
```
cd agents
python agent_runner_copy.py --agent budget_agent
```

### Supervisor Agent (TERMINAL 6)

```
cd agents
python agent_runner_copy.py --agent travel_router
```

# Start UI (TERMINAL 7)

```
python start_ui.py
```

Go to http://127.0.0.1:7860

# Customization
You can add any agents by configuring the yaml files

TBC