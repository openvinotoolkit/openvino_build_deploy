<div align="center">

# Agentic Tourism - Multi-Agent Travel Assistant with OpenVINO™ Toolkit  
A sophisticated multi-agent system that provides intelligent travel assistance using specialized AI agents for hotel search, flight booking, and image captioning travel recommendations. Built with OpenVINO, MCP (Model Context Protocol), A2A (Agent to Agent Protocol), and the BeeAI framework.

  <h4>
    <a href="https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/open-potential.html">🏠&nbsp;About&nbsp;the&nbsp;Kits&nbsp</a>
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

- [OpenVINO/Qwen3-8B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) (LLM)
- [OpenVINO/Phi-3.5-vision-instruct-int4-ov](https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int4-ov) (VLM)

Check out our [AI Reference Kits repository](https://github.com/openvinotoolkit/openvino_build_deploy) for other kits.

```
┌─────────────────────────────────────────────────────────┐
│                    Travel Router Agent                  │
│              (Main Coordinator & Orchestrator)          │
│                        (Port 9996)                      │
└────────────────────────┬────────────────────────────────┘
                         │
       ┌───────────-─────┼─────────────────┐
       │                 │                 │             
       ▼                 ▼                 ▼           
┌─────────────┐   ┌─────────────┐   ┌─────────────┐ 
│Hotel Finder │   │Flight Finder│   │Image Proc.  │ 
│   Agent     │   │   Agent     │   │   Agent     │ 
│  (Port 9999)│   │ (Port 9998) │   │ (Port 9997) │ 
└─────────────┘   └─────────────┘   └─────────────┘ 
       │                 │                 │               
       ▼                 ▼                 ▼              
┌─────────────┐   ┌─────────────┐   ┌─────────────┐ 
│Hotel Search │   │Flight Search│   │Image Caption│ 
│ MCP Server  │   │ MCP Server  │   │  MCP Server │
│ (Port 3001) │   │ (Port 3002) │   │ (Port 3003) │
└─────────────┘   └─────────────┘   └─────────────┘ 
                          │
                          ▼
┌────────────────────────────────────────────────────────────┐
│             Hardware OpenVINO AI Stack                     │
│       • Qwen3-8B-int4-ov (LLM)                             │
│       • Phi-3.5-vision-instruct-int4-ov (VLM)              │
└────────────────────────────────────────────────────────────┘

```

---

## Steps
1. Start OVMS LLMs
2. Start MCP servers (`start_mcp_servers.py`)
3. Start Agents      (`start_agents.py`)
4. Start UI          (`start_ui.py`)

## Setting Up Your Environment

To set up your environment, you first clone the repository, then create a virtual environment, activate the environment, and install the packages.

### Clone the Repository

To clone the repository, run this command:

```shell
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git
```

This command clones the repository into a directory named `openvino_build_deploy` in the current directory. After the directory is cloned, run the following command to go to that directory:


```shell
cd openvino_build_deploy/ai_ref_kits/agentic_multimodal_travel_planer
```

### Create virtual environment

```
python3 -m venv agentic_venv
```

### Activate the Environment

The command you run to activate the virtual environment you created depends on whether you have a Unix-based operating system (Linux or macOS) or a Windows operating system.

To activate  the virtual environment for a **Unix-based** operating system, run:

```shell
source agentic_venv/bin/activate  
```

To activate the virtual environment for a **Windows** operating system, run:

```shell
agentic_venv\Scripts\activate
```

This activates the virtual environment and changes your shell's prompt to indicate that you are now working in that environment.

### Install the Requirements

To install the required packages, run the following commands:

```shell
python -m pip install --upgrade pip 
pip install -r requirements.txt
```

## Step 1: Getting the LLMs/VLM for agents ready with OpenVINO model Server (OVMS)

### OPTION 1: Linux (Docker)

#### Docker Installation
For installation instructions, refer to the [official Docker documentation for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

#### Download images and Run models
Once you have Docker installed on your machine, run the following script which will download the images and start the containers:
```
chmod +x download_and_run_models_linux.sh
./download_and_run_models_linux.sh
```

The script PULL the models from Hugging Face and will start the docker containers for you. It might take few minutes, please wait until you see the following confirmation.
```
==============================================
OpenVINO Model Server is running on:
----------------------------------------------
LLM : http://localhost:8001
VLM : http://localhost:8002
==============================================
```

### Verify the services are running

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
### Customization options 
The script also provides a stop command which will stop and remove the models containers:
```
./download_and_run_models_linux.sh --stop
```
You have the option to configure different models.  
**NOTE**  The peformance may vary depending on the size of the model you select. We HIGHLY recommend using OpenVINO optimized models from the [OpenVINO Hugging Face public repository](https://huggingface.co/OpenVINO/models).
```
  # Use different models
  ./download_and_run_models_linux.sh -llm-model "OpenVINO/Llama-3.1-8B-int4-ov" --vlm-model "OpenVINO/LLaVA-NeXT-7B-int4-ov"

  # Use different ports
  ./download_and_run_models_linux.sh--llm-port 9001 --vlm-port 9002
```

### OPTION 2: Windows (Binary)

#### Download images and Run models using Binaries
Run the following command:

```
./download_and_run_models_Windows.bat
```

NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) 

## Step 2: Start the MCP servers

This example uses three MCP servers that the agents will consume:
- Flight search (Intel AI Builder): provides available flight options
- Hotel search (Intel AI Builder): provides available hotels
- Image captioning: generates captions for images

### Get your SerpAPI key
The flight and travel agents rely on an external API to search for flights and hotels. To enable this, you’ll need an API key from SerpAPI.

You can sign up for a free API key, which includes 250 requests per month and does not require a credit card. You’ll only need to upgrade if you plan to exceed the free monthly limit.

1. Go to https://serpapi.com/
2. Navigate to "Your Private API Key"
3. Copy the key

Once you have your key, you can launch the MCP servers.

### Launch MCP servers

Run and note you will be required to add the KEY
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
```
The script also provides a stop command:

```
python start_mcp_servers.py --stop
```

## Step 3: Start Agents

Start all the agents.

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

Logs are in `logs/`. You can open each agent's log file there.

The script also provides a stop command:

```
python start_agents.py --stop
```

Logs are in `logs/`. You can navigate to the folder to the log of each Agent server.

## Step 4: Start UI

```
python start_ui.py
```

Open `http://127.0.0.1:7860` in your browser.

## Customization (optional)

### Agents configuration
Below is how you can configure your own agents for the Agentic Multimodal Travel Planner. This guide covers the basics of agent configuration and helps you set up new agents or customize existing ones.

#### 1. Overview

Agents are defined via YAML configuration files located in the `config/` directory:

- `config/agents_config.yaml`: Controls properties, endpoints, ports, and enabled status of each agent.
- `config/agents_prompts.yaml`: Defines the system instructions (prompt) and templates for communication of each agent.

#### 2. How to add a new agent

1. **Open `config/agents_config.yaml`**  
   Find an agent section (e.g., `flight_finder:`), and use it as a template for your new agent.

   ```yaml
   your_custom_agent:
     name: "agent_name"
     port: 9988
     role : "agent_role"
     enabled: true
   ```

   - `your_custom_agent`: Unique name for your agent.
   - `name`: Human-readable identifier for your agent.
   - `port`: TCP port the agent will listen on (make sure it's unused).
   - `role`: Brief role/description for your agent.
   - `enabled`: Set to `true` to enable the agent.

2. **Open `config/agents_prompts.yaml`**  
   Create an entry for your agent with its system prompt and dialogue template.

   ```yaml
   your_custom_agent:
     system: |
       You are the Custom Agent. Your job is to provide ... (describe the behavior here)
     template: |
       [User]: {{query}}
       [Agent]: 
   ```

   - The `system` prompt is injected as the agent’s system-level context.
   - The `template` can be customized to guide your agent's responses.

4. **Start Your Agent**  
   Make sure your agent is enabled in `agents_config.yaml`, then run:

   ```
   python start_agents.py
   ```

   Your agent should start alongside the others. Check the logs in the `logs/` directory for messages from your agent.

#### 3. Tips for Customization

- Use unique port numbers for each new agent.
- You may copy config stanzas for existing agents as a quick start.
- Edit the system prompt to set the "personality" and role for your agent.
- Restart the agent process after changing any YAML config.

#### 4. Disabling/Enabling Agents

Set `enabled: false` to disable an agent in `config/agents_config.yaml`—that agent won't be started by `start_agents.py`.

#### 5. Troubleshooting

- If your agent doesn't appear, check the logs in `logs/` for errors.
- Make sure the port is not being used by another process.
- Ensure your new agent class is properly importable and subclassed as required by the BeeAI Framework.

---

With this approach, you can flexibly expand the capabilities of your travel planning system by adding or customizing new agents to fit your requirements!

[//]: # (telemetry pixel)
<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=7003a37c-568d-40a5-9718-0d021d8589ca&project=ai_ref_kits/agentic_multimodal_travel_planner&file=README.md" />
