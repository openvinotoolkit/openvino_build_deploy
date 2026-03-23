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

## Quick Start

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

### Linux (Docker)

Use the launcher:

```bash
chmod +x run_all_linux.sh
./run_all_linux.sh
```

Stop everything:

```bash
./run_all_linux.sh --stop
```

The kit ships with tested default models and runs without extra flags. The options below are optional if you want to customize models, ports, or device selection:

```bash
./run_all_linux.sh --device CPU --llm-port 9001 --vlm-port 9002
./run_all_linux.sh --llm-model "OpenVINO/Llama-3.1-8B-int4-ov" --vlm-model "OpenVINO/LLaVA-NeXT-7B-int4-ov"
./run_all_linux.sh --llm-device GPU.0 --vlm-device CPU
```

### Windows (Binary)

Use the launcher:

```bat
run_all_windows.bat
```

Stop everything:

```bat
run_all_windows.bat --stop
```

The kit ships with tested default models and runs without extra flags. The options below are optional if you want to customize models, ports, or device selection:

```bat
run_all_windows.bat --device CPU --llm-port 9001 --vlm-port 9002
run_all_windows.bat --llm-model "OpenVINO/Llama-3.1-8B-int4-ov" --vlm-model "OpenVINO/LLaVA-NeXT-7B-int4-ov"
run_all_windows.bat --llm-device GPU.0 --vlm-device CPU
```

> NOTE: If you are using Windows, you may need to install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)

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
