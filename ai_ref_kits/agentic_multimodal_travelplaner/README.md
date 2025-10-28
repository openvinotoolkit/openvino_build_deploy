<div align="center">

# Agentic Tourism - Multi-Agent Travel Assistant with OpenVINO™ Toolkit  
A sophisticated multi-agent system that provides intelligent travel assistance using specialized AI agents for hotel search, flight booking, and image captioning travel recommendations. Built with OpenVINO, MCP (Model Context Protocol), A2A (Agent to Agent Protocol), and the BeeAI framework.

  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp;·</a>
  </h4>
</div>

[![Apache License](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_build_deploy/blob/master/LICENSE.txt)

---

This reference kit demonstrates a multi-agent travel assistant. It coordinates specialized agents for hotel and flight search via MCP-connected tools and uses an image captioning VLM for visual understanding. The system is built with OpenVINO™ and the OpenVINO Model Server for optimized local inference, and orchestrated using BeeAI, MCP, and the A2A protocol.

This kit uses the following technology stack:

- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [OpenVINO™ GenAI](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html)
- [Optimum Intel](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html)
- [OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_what_is_openvino_model_server.html)
- [BEE AI Framework](https://github.com/i-am-bee/beeai-framework)
- [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)
- [A2A](https://github.com/a2aproject/A2A)

- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (LLM)
- [OpenVINO/Phi-3.5-vision-instruct-int4-ov](https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int4-ov) (VLM)

Check out our [AI Reference Kits repository](https://github.com/openvinotoolkit/openvino_build_deploy) for other kits.

```
┌─────────────────────────────────────────────────────────┐
│                    Travel Router Agent                  │
│              (Main Coordinator & Orchestrator)          │
│                        (Port 9996)                      │
└─────────────────────────┬───────────────────────────────┘
                          │
   ┌───────────────-──────┼────────────────┐
   │                      │                │             
   ▼                      ▼                ▼           
┌─────────────┐   ┌─────────────┐   ┌─────────────┐ 
│Hotel Finder │   │Flight Finder│   │Image Proc.  │ 
│   Agent     │   │   Agent     │   │   Agent     │ 
│  (Port 9999)│   │ (Port 9998) │   │ (Port 9995) │ 
└─────────────┘   └─────────────┘   └─────────────┘ 
      │                 │                 │               
      ▼                 ▼                 ▼              
┌─────────────┐   ┌─────────────┐ ┌─────────────┐ 
│Hotel Search │   │Flight Search│ │Image Caption│ 
│ MCP Server  │   │ MCP Server  │ │  MCP Server │
│ (Port 3001) │   │ (Port 3002) │ │ (Port 3003) │
└─────────────┘   └─────────────┘ └─────────────┘ 
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│             Hardware OpenVINO AI Stack                     │
│       • Qwen3-8B-int4-ov (LLM)                             │
│       • Phi-3.5-vision-instruct-int4-ov (VLM)              │
└────────────────────────────────────────────────────────────┘

```

---

# STEPS
- Start OVMS LLMs
- Start MCP servers (start_mcp_servers.py)
- Start Agents      (start_agents.py)
- Start UI           (start_ui.py)


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

## Create virtual environment

```
python3 -m venv agentic
```

### Activate the Environment

The command you run to activate the virtual environment you created depends on whether you have a Unix-based operating system (Linux or macOS) or a Windows operating system.

To activate  the virtual environment for a **Unix-based** operating system, run:

```shell
source agentic/bin/activate  
```

To activate the virtual environment for a **Windows** operating system, run:

```shell
agentic\Scripts\activate
```

This activates the virtual environment and changes your shell's prompt to indicate that you are now working in that environment.

### Install the Requirements

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

# Step 1: Getting the LLM for agents ready with OpenVINO model Server (OVMS) using optimized models

Windows and Linux
## OPTION 1: Windows

TBC

## OPTION 2: Linux

### Docker Installation
For instalation instructions details, refer to the [official Docker documentation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

### Get OpenVINO Model Server image
Once you have Docker installed on your machine, pull the OpenVINO Model Server image:
```
docker pull openvino/model_server:latest
```

### Download optimized models 

OpenVINO Model Server will serve your models. In this example you will use two models: an LLM and a VLM.

Create your folder 
```
sudo mkdir -p $(pwd)/models
sudo chown -R $(id -u):$(id -g) $(pwd)/models
chmod -R 755 $(pwd)/models   
```

Agent LLM: **Qwen3-8B**
```
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models openvino/model_server:latest --pull --model_repository_path /models --source_model OpenVINO/Qwen3-8B-int4-ov --task text_generation --tool_parser hermes3
```

Vision Language Model (VLM): **Phi-3.5-vision-instruct-int4-ov**
```
docker run --user $(id -u):$(id -g) --rm -v $(pwd)/models:/models:rw openvino/model_server:latest  \
--pull --model_repository_path /models --source_model OpenVINO/Phi-3.5-vision-instruct-int4-ov --task text_generation --pipeline_type VLM
```

### Start OpenVINO Model Server 
Once you have your models, your next step is to start the services. 

LLM
```
docker run -d --user $(id -u):$(id -g) --rm \
  -p 8001:8000 \
  -v $(pwd)/models:/models openvino/model_server:latest \
  --rest_port 8000 \
  --model_repository_path /models \
  --source_model OpenVINO/Qwen3-8B-int4-ov \
  --tool_parser hermes3 \
  --cache_size 2 \
  --task text_generation \
  --enable_prefix_caching true
```


VLM
```
docker run -d --rm \
  -p 8002:8000 \
  -v $(pwd)/models:/models:ro \
  openvino/model_server:latest \
  --rest_port 8000 \
  --model_name OpenVINO/Phi-3.5-vision-instruct-int4-ov \
  --model_path /models/OpenVINO/Phi-3.5-vision-instruct-int4-ov
```

### Check
Run:
```
docker ps
```

You should see the two models serving 
```
CONTAINER ID   IMAGE                          COMMAND                  CREATED       STATUS       PORTS                                         NAMES
424634ea10fe   openvino/model_server:latest   "/ovms/bin/ovms --re…"   3 days ago    Up 3 days    0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp   competent_ganguly9
a962a7695b1f   openvino/model_server:latest   "/ovms/bin/ovms --re…"   3 days ago    Up 3 days    0.0.0.0:8002->8000/tcp, [::]:8002->8000/tcp   agitated_galois
```

Your LLM is now running and ready to be used by the agents.

# Step 2: Start the Agents and the MCP tools

## Start MCP tools
This example uses three MCP servers that the agents will consume:
- Flight search (Intel AI Builder): provides available flight options
- Hotel search (Intel AI Builder): provides available hotels
- Image captioning: generates captions for images

## Get your SerpAPI key
Flight and travel agents use an external API for hotel and flight search. Obtain an API key from SerpAPI.

Go to https://serpapi.com/

Navigate to "Your Private API Key"

Once you have your key, you can launch the MCP servers.

## Launch MCP servers

Set your key
```
export SERP_API_KEY=***YOUR_KEY***
```

Run
```
python start_mcp_servers.py
```

**NOTE**: This script starts the MCP servers in the background and reads configuration from `config/mcp_config.yaml`. You can configure each MCP server there.

You should see confirmation that the MCP servers are running:
```
MCP 'image_mcp' started on port 3003
MCP 'hotel_finder' started on port 3001
MCP 'flight_finder' started on port 3002

Successfully started MCP servers: image_mcp, hotel_finder, flight_finder

Logs are in `logs/`. You can open each MCP server's log file there.

The script also provides a stop command:

```
python start_mcp_servers.py --stop
```
# Step 3: Start Agents

Start all agents
```
python start_agents.py
```

You should see:
```
Agent 'travel_router' started on port 9996
Agent 'flight_finder' started on port 9998
Agent 'hotel_finder' started on port 9999
Agent 'image_captioning' started on port 9997

Successfully started agents: travel_router, flight_finder, hotel_finder, image_captioning

Logs are in `logs/`
```
**NOTE**: To inspect individual agent behavior, start each agent in its own terminal with `python agents/agent_runner.py --agent AGENT_NAME`.

Logs are in `logs/`. You can navigate to the folder to the log to each Agent server


# Step 4: Start UI

```
python start_ui.py
```

Open `http://127.0.0.1:7860` in your browser.

# OPTIONAL Customization
You can add agents by configuring the YAML files

## Agents YAML config

All agents are created by a runner `agents/agent_runner.py` which reads two files: `config/agents_config.yaml` and `config/agents_prompt.yaml`.

You can start the agents all together by running 
```python
python start_agents.py
```

### Agents configuration

### Agents prompts