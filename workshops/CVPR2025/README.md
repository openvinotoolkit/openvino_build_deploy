
# AgenticWorkflow 

## Overview
AgenticWorkflow is a multimodal AI system that combines video search and shopping cart functionality through MCP (Model Context Protocol) servers and a Gradio-based web interface.
(Recommended Windows laptop with 32GB RAM)

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Technical Implementation Details](#technical-implementation-details)
- [Setup Guide](#setup-guide)
- [Strengths](#strengths)
- [Areas for Enhancement](#areas-for-enhancement)
- [Conclusion](#conclusion)

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio Web    │    │  Multi-Agent     │    │  MCP Servers    │
│   Interface     │◄──►│  Workflow        │◄──►│                 │
│ (gradio_helper) │    │                  │    │ 1. Video Search │
└─────────────────┘    └──────────────────┘    │ 2. Shopping Cart│
                                               │ 3. Realtime Cap │
                                               └─────────────────┘
```

## Core Components

### 1. **MCP Servers** (Model Context Protocol)

#### A. Video Processing Server (`video_processing_server.py`)
- **Port**: 3000 (localhost:3000/sse)
- **Purpose**: Multimodal video search and analysis
- **Key Functions**:
  - `ingest_videos()` - Process and index video content
  - `search_from_video()` - Search video frames using text queries

<!-- **Technical Stack**:
- **BridgeTower** embeddings for multimodal text-vision understanding
- **OpenVINO Whisper** for audio transcription 
- **LanceDB** vector database for similarity search
- **VLM (Vision Language Model)** for frame analysis
- **OpenCV** for frame extraction -->

- **Workflow**:
    1. Accept base64-encoded video files
    2. Extract audio and generate transcripts using Whisper
    3. Extract video frames at transcript intervals
    4. Generate multimodal embeddings using BridgeTower
    5. Store in LanceDB vector database
    6. Process search queries against indexed content
    7. Return both text responses and base64-encoded frame images

#### B. Shopping Cart Server (`shopping_cart_server.py`)
- **Port**: 3001 (localhost:3001/sse)
- **Purpose**: E-commerce functionality with product search and cart management
- **Key Functions**:
  - `product_query()` - Search paint products and get recommendations
  - `add_to_cart()` - Add items to shopping cart
  - `view_cart()` - Display cart contents
  - `clear_cart()` - Empty the cart
  - `calculate_paint_cost()` - Calculate project costs
  - `calculate_paint_gallons()` - Calculate paint quantity needed

<!-- **Technical Stack**:
- **BGE Large EN v1.5** embeddings for product search
- **LlamaIndex** for document indexing and retrieval
- **PDF document** (`test_painting_llm_rag.pdf`) as product knowledge base
- In-memory cart storage using Python lists -->

#### C. Real-time Capture Server (`realtime_capture_server.py`) **[NEW]**
- **Port**: 3002 (localhost:3002/sse)
- **Purpose**: Live webcam and microphone capture with multimodal analysis
- **Key Functions**:
  - `start_realtime_capture()` - Begin live capture from webcam and microphone
  - `stop_realtime_capture()` - Stop capture and release resources
  - `process_and_store_recent_data()` - Process recent audio-visual data
  - `search_realtime_data()` - Search through captured real-time content
  - `get_capture_status()` - Monitor system status
<!-- 
**Technical Stack**:
- **OpenCV** for webcam frame capture
- **SoundDevice** for microphone audio recording
- **OpenVINO Whisper** for real-time audio transcription
- **BridgeTower** embeddings for live multimodal understanding
- **Threading** for concurrent audio/video capture
- **LanceDB** vector database for real-time content indexing -->
- **Workflow**:
    1. Capture live video frames from webcam (configurable FPS)
    2. Record audio segments from microphone (configurable duration)
    3. Transcribe audio using OpenVINO Whisper
    4. Align video frames with corresponding audio transcripts
    5. Generate BridgeTower embeddings for each frame-text pair
    6. Store in vector database for real-time search
    7. Enable live query and analysis of captured content

### 2. **Multi-Agent Workflow Client** (`multiagents_workflow.py`)

The core orchestration layer that manages multiple AI agents:

**Agents**:
- **RouterAgent**: Analyzes queries and routes to appropriate specialist agent
- **VideoSearchAgent**: Handles video-related queries using video processing MCP tools
- **ShoppingCartAgent**: Manages e-commerce queries using shopping cart MCP tools

**Key Features**:
- **Intelligent Routing**: Automatically determines which agent should handle each query
- **MCP Response Handling**: Complex callback system to handle fragmented responses
- **Memory Management**: Maintains conversation context across agent interactions
- **Stream Processing**: Real-time response streaming for better user experience

### 3. **Gradio Web Interface** (`gradio_helper.py`)

**Features**:
- **Interactive Chat**: Real-time conversation with multi-agent system
- **Video Upload**: Direct video ingestion through web interface
- **Shopping Cart Display**: Live cart updates in sidebar
- **Agent Step Visualization**: Shows which agents are being used
- **Example Queries**: Pre-built queries to demonstrate functionality

**UI Components**:
- Main chat interface for user interaction
- Agent steps panel showing workflow progression
- Shopping cart panel with live updates
- Video upload component
- Example query buttons


## Technical Implementation Details


### AI Models Used

1. **Phi-4 Multimodal Instruct** (VLM)
   - INT4 quantized for efficiency
   - Handles image-text reasoning
   - Device: GPU.0

2. **Qwen2.5-7B-Instruct** (LLM) 
   - INT4 quantized text generation
   - Powers agent reasoning and responses
   - Device: GPU.0

3. **BGE Large EN v1.5** (Embeddings)
   - INT4 quantized embeddings
   - Used for product search and retrieval
   - Device: CPU

4. **BridgeTower** (Multimodal Embeddings)
   - Custom OpenVINO implementation
   - Combines text and vision understanding
   - Device: GPU.0

5. **OpenVINO Whisper** (Speech Recognition)
   - Audio transcription from videos
   - Optimized for Intel hardware

### Key Python Scripts

1. **`main_search_server.py`**: Starts video processing MCP server
2. **`main_shopping_cart_server.py`**: Starts shopping cart MCP server  
3. **`main_realtime_capture_server.py`**: Starts real-time capture MCP server **[NEW]**
4. **`gradio_helper.py`**: Launches web interface

<!-- #### Client Components

1. **`video_ingestion_client.py`**: Handles video upload and processing
2. **`multiagents_workflow.py`**: Core agent orchestration
3. **`working_video_search_client.py`**: Standalone video search client
4. **`main_search_client.py`**: Command-line MCP client
5. **`realtime_capture_client.py`**: Real-time capture demonstration client **[NEW]**

#### Utility Modules

1. **`src/utils/`**: 
   - `custom_embeddings.py` - OpenVINO embedding implementations
   - `logger.py` - Logging utilities
   - `utils.py` - Common helper functions

2. **`src/mcp_servers/bridgetower_search/utils/`**:
   - `utils.py` - Video processing utilities
   - Frame extraction, transcript processing, vector store operations

3. **`src/mcp_servers/shopping_cart/utils.py`**:
   - Document loading and model setup
   - OpenVINO LLM configuration

4. **`src/mcp_servers/realtime_capture/`** **[NEW]**:
   - `realtime_capture_server.py` - Live webcam and microphone processing
   - `README.md` - Detailed documentation for real-time capture system -->

<!-- ## Data Flow -->

<!-- ### Video Search Flow
```
User Query → RouterAgent → VideoSearchAgent → Video MCP Server
                                            ↓
Base64 Video → Frame Extraction → Whisper Transcription
                                            ↓  
BridgeTower Embeddings → LanceDB Storage → Similarity Search
                                            ↓
VLM Analysis → JSON Response → Agent Callback → User Interface
```

### Shopping Cart Flow
```
User Query → RouterAgent → ShoppingCartAgent → Shopping MCP Server
                                             ↓
Product Query → BGE Embeddings → Document Search → Product Info
                                             ↓
Cart Operations → In-Memory Storage → JSON Response → User Interface
```

### Real-time Capture Flow **[NEW]**
```
Webcam Frames + Microphone Audio → Real-time Capture Manager
                                            ↓
OpenVINO Whisper Transcription → Frame-Text Alignment → BridgeTower Embeddings
                                            ↓
LanceDB Storage → Similarity Search → VLM Analysis → Live Query Results
``` -->




## Setup Guide


###  Environment Setup

```bash
Install Python 3.12
git lfs install
git clone https://github.com/openvinotoolkit/openvino_build_deploy.git 
cd openvino_build_deploy/workshops/CVPR2025/
python -m venv env
env/scripts/activate

pip install -r requirements.txt
choco install ffmpeg
```

### Export OpenVINO Models

Run the following commands to prepare the required AI models:

#### Phi-4 Multimodal Instruct Model
```bash
optimum-cli export openvino --model microsoft/Phi-4-multimodal-instruct phi-4-multimodal-instruct/INT4 --trust-remote-code --task image-text-to-text --weight-format int4 --group-size 64
```

#### Qwen2.5-7B-Instruct Model
```bash
optimum-cli export openvino -m Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct/INT4 --trust-remote-code --task text-generation-with-past --weight-format int4 --sym --ratio 1.0 --group-size 128
```

#### BGE Large EN v1.5 Embedding Model
```bash
optimum-cli export openvino --model BAAI/bge-large-en-v1.5 bge-large-en-v1.5-dyn-int4/INT4 --task feature-extraction --trust-remote-code --framework pt --library sentence_transformers --weight-format int4 --group-size 128 --sym --ratio 1
```

#### Whisper Model Conversion
```bash
python convert_whisper_model.py
copy whisper_tokenizers\* files to whisper model folder
(These tokenizers/detokenizers are pre-generated by method in https://docs.openvino.ai/2025/openvino-workflow-generative/ov-tokenizers.html)
```
Then move the whisper-models folder under the main models folder

#### BridgeTower Model
```bash
cd openvino_bridgetower
$env:TOKENIZERS_PARALLELISM="false"; python openvino_bridgetower_conversion.py
```
Then move the bridgetower_models folder under the main models folder

### Environment Variables (`src/.env`)
- Model paths for all AI components
- MCP server URLs and configurations  
- Vector database settings
- Device assignments (GPU/CPU)
- Document paths

Create and configure your `src/.env` file 

> **Note**: Use regular whisper model instead of OpenVINO whisper for now

### 🖥️ Running the Application

### 1. Start MCP Servers

Run the following servers in separate terminal windows:

```bash
cd src

# Shopping Cart MCP Server
python main_shopping_cart_server.py


Log as below:
LLM is explicitly disabled. Using MockLLM.
INFO:     Started server process 
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:3001 (Press CTRL+C to quit)

# Search Server  
python main_search_server.py

Log as below:
🔧 BridgeTower Model Configuration:
   📱 Device: GPU.0
   📄 Text-Vision Model: bridgetower_large_itc.xml
   📄 Text Model: custombridgetower_text_large_itc.xml
   ✅ Text-Vision Model compiled on GPU.0
   ✅ Text Model compiled on GPU.0
INFO:     Started server process 
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:3000 (Press CTRL+C to quit)
```

### 2. Launch Web Browser Client

```bash
# Set temporary directory for Gradio
set GRADIO_TEMP_DIR='C:\working\CVPR2025\gradio_tmp'


# Start the web interface
python gradio_helper.py
```

### 📝 Notes

- Ensure all model paths in the `.env` file point to the correct locations on your system
- GPU devices can be specified as `GPU.0`, `GPU.1`, etc., or use `CPU` for CPU inference
- The Gradio temporary directory should be set to a location with sufficient disk space


## Strengths

1. **Modular Architecture**: Clean separation between servers, agents, and interface
2. **Multimodal Capabilities**: Combines text, image, and video understanding
3. **GPU Optimization**: Efficient inference with OpenVINO and INT4 quantization
4. **Windows Compatibility**: Windows compatible
5. **User-Friendly Interface**: Intuitive Gradio web interface
6. **Flexible Agent System**: Easy to extend with new agents and capabilities

## Areas for Enhancement

1. **Persistent Storage**: Shopping cart currently uses in-memory storage
2. **User Authentication**: No user management system
3. **Scalability**: Single-user design, could benefit from multi-user support
4. **Error Handling**: Could use more robust error recovery mechanisms
5. **Testing**: Limited automated testing coverage
6. **Documentation**: Could benefit from API documentation
7. **Real-time Performance**: Resource-intensive processing may need optimization **[NEW]**
8. **Data Retention**: Real-time capture needs configurable storage policies **[NEW]**

## Conclusion

AgenticWorkflow represents a sophisticated multimodal AI system that successfully combines video understanding with e-commerce functionality. The Windows port demonstrates good engineering practices with proper path handling, GPU optimization, and robust response processing. The modular MCP architecture makes it easy to extend and maintain, while the multi-agent design provides flexible query routing and specialized task handling.

**NEW**: The addition of real-time capture capabilities extends the system's utility to live interactions, making it suitable for demonstrations, meetings, educational applications, and interactive content creation. The system now supports three complementary modes: batch video processing, e-commerce interactions, and live audio-visual capture and analysis.