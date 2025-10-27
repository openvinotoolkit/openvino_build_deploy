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
- []() (VLM)

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
│ (Port 7901) │   │ (Port 7902) │ │ (Port 7903) │
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
- Start MCP servers (start_mcp.py)
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

# Step 1: Getting the LLM for agents ready with OpenVINO model Server (OVMS) using optimized models

Windows and Linux
## OPTION 1: Windows

TBC

## OPTION 2: Linux

### Docker Installation
To install Docker on Ubuntu, follow these steps:

1. **Update your existing list of packages:**
   ```shell
   sudo apt-get update
   ```

2. **Install packages to allow apt to use a repository over HTTPS:**
   ```shell
   sudo apt-get install \
     ca-certificates \
     curl \
     gnupg
   ```

3. **Add Docker’s official GPG key:**
   ```shell
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   sudo chmod a+r /etc/apt/keyrings/docker.gpg
   ```

4. **Set up the Docker repository:**
   ```shell
   echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. **Update the apt package index:**
   ```shell
   sudo apt-get update
   ```

6. **Install Docker Engine, containerd, and Docker Compose:**
   ```shell
   sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

7. **Verify that Docker Engine is installed correctly:**
   ```shell
   sudo docker run hello-world
   ```

For more details, refer to the [official Docker documentation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

### Get OpenVINO Model Server image
Once you have Docker installed on your machine, pull the OpenVINO Model Server image:
```
docker pull openvino/model_server:latest
```

### Download optimized models 

OpenVINO Model Server will serve your models. In this example you will use two models: an LLM and a VLM.

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
  --model_repository_path models \
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

## Clone MCP servers from AI Builder
```
cd mcp_tools
wget -O /home/ubuntu/openvino_build_deploy/workshops/Agentic_Tourism_copy/mcp_tools/ai_builder_mcp_flight_finder.py \
https://raw.githubusercontent.com/intel/intel-ai-assistant-builder/main/mcp/mcp_servers/mcp_google_flight/server.py

```

## Start MCP tools
In this example, we will have multiple MCP servers to be consumed by the agents, there will be 3 MCP tools
- Flight_search : It servers to provide avaialble flights 
- Hotel_search : It servers to provide available hotels
- Image Captioning : Caption the image

## Get your SerpAPi
Flight and Travel agents use an external API that provides searches from hotels and flights. This is provided by serapo, please be sure to get your key.

Go to https://serpapi.com/

Navigate to "Your Private API Key"

Once you have your key 

Time to launch the MCP servers

## Launch MCP servers

Set your key
```
export SERP_API_KEY=***YOUR_KEY***
```

Run
```
start_mcp_servers.py
```

**NOTE**: This script starts the MCP servers in the backgroud reads the configuration set on `/config/mcp_config.yaml`. There you can set each MCP parameter

You should get a confirmation that the MCP servers are up and running
```
MCP 'image_mcp' started on port 3005
MCP 'hotel_finder' started on port 7903
MCP 'flight_finder' started on port 7901

Successfully started MCP servers: image_mcp, hotel_finder, flight_finder

Logs are in logs/
```

The script also provides a stop function to stop them: 
```
python start_mcp_servers.py --stop
```

## Agents YAML config

All agents are created by a runner `agents/agent_runner.py` which reads two files: `config/agents_config.yaml` and `config/agents_prompt.yaml`.

You can start the agents all together by running 
```python
python start_agents.py
```

### Agents configuration

### Agents prompts





# Step 3: Start Agents

Start all agents
```
python start_agents.py
```


### Regular A2A

Hotel Finder agent (TERMINAL 3)

```
cd agents
python agent_runner.py --agent hotel_finder
```

Flight Finder agent (TERMINAL 4)
```
cd agents
python agent_runner.py --agent flight_finder
```


Budget Approval agent (TERMINAL 5)
```
cd agents
python agent_runner.py --agent budget_agent
```

### Supervisor agent (TERMINAL 6)

```
cd agents
python agent_runner.py --agent travel_router
```

# Step 4 Start UI

```
python start_ui.py
```

Open `http://127.0.0.1:7860` in your browser.

# OPTIONAL Customization
You can add agents by configuring the YAML files

TBC