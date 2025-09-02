# Real-time Capture System

## Overview
This folder contains the real-time capture system that extends the AgenticWorkflow project to support live webcam video and microphone audio processing using BridgeTower embeddings.

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Webcam     │    │  Real-time      │    │  BridgeTower     │
│      +       │───►│  Capture        │───►│  Embeddings      │
│  Microphone  │    │  Manager        │    │                  │
└──────────────┘    └─────────────────┘    └──────────────────┘
                            │                        │
                            ▼                        ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Gradio     │    │   MCP Server    │    │   LanceDB        │
│   Client     │◄───│   (Port 3002)   │◄───│   Vector Store   │
└──────────────┘    └─────────────────┘    └──────────────────┘
```

## Key Features

### 1. **Live Data Capture**
- **Webcam**: Captures video frames at configurable frame rates (default: 2 fps)
- **Microphone**: Records audio segments in configurable durations (default: 5 seconds)
- **Synchronized Processing**: Aligns video frames with corresponding audio segments

### 2. **Real-time Processing**
- **OpenVINO Whisper**: Transcribes audio segments to text
- **BridgeTower Embeddings**: Generates multimodal embeddings from frame+text pairs
- **Vector Storage**: Stores embeddings in LanceDB for similarity search

### 3. **MCP Server Interface**  
- **Port 3002**: Dedicated server for real-time capture functionality
- **SSE Transport**: Server-Sent Events for real-time communication
- **Tool Integration**: Compatible with existing MCP client infrastructure

## Components

### `realtime_capture_server.py`
**Main server implementation with the following tools:**

#### Tools Available:
1. **`start_realtime_capture(frame_rate, audio_duration)`**
   - Starts webcam and microphone capture
   - Configurable frame rate and audio segment duration
   - Creates background threads for continuous capture

2. **`stop_realtime_capture()`**
   - Stops all capture processes
   - Releases camera and audio resources
   - Cleans up background threads

3. **`process_and_store_recent_data(frame_count, audio_count)`**
   - Processes recent captured frames and audio
   - Generates transcripts using OpenVINO Whisper
   - Creates BridgeTower embeddings and stores in vector DB
   - Aligns frames with audio by timestamp

4. **`search_realtime_data(query, top_k)`**
   - Searches stored real-time data using text queries
   - Returns both text analysis and matching images
   - Uses VLM for intelligent result analysis

5. **`get_capture_status()`**
   - Returns current system status
   - Buffer sizes, queue states, camera availability

### `RealtimeCaptureManager` Class
**Core capture management:**
- **Thread Management**: Separate threads for video and audio capture
- **Buffer Management**: Circular buffers for recent frames and audio
- **Queue System**: Thread-safe queues for data processing
- **Resource Management**: Proper cleanup of camera and audio resources

### `main_realtime_capture_server.py`
**Server entry point:**
- Starts the MCP server on port 3002
- Simple launch script for the real-time capture system

### `realtime_capture_client.py`
**Client demonstration:**
- Interactive CLI for testing server functionality
- Commands for start/stop, process, search, and status
- Example usage patterns

## Technical Details

### Data Flow
```
Webcam Frame → Base64 Encoding → Frame Buffer → Processing Queue
      ↓                                              ↓
Microphone → WAV File → Whisper → Transcript → Text-Image Pairs
      ↓                                              ↓
BridgeTower Embeddings → Vector Storage → Search Index
```

### Threading Architecture
- **Main Thread**: MCP server and request handling
- **Frame Capture Thread**: Continuous webcam capture at specified FPS
- **Audio Capture Thread**: Continuous microphone recording in segments
- **Processing**: Synchronous processing when triggered by MCP calls

### Memory Management
- **Frame Buffer**: Circular buffer (max 100 frames) for recent video data
- **Audio Buffer**: Circular buffer (max 20 segments) for recent audio data
- **Queues**: Limited-size queues prevent memory overflow
- **Cleanup**: Automatic cleanup of temporary files and old data

## Dependencies

### New Dependencies Required:
```bash
pip install sounddevice  # Audio capture
pip install opencv-python  # Already included in project
pip install webvtt-py  # Already included in project
```

### Existing Dependencies Used:
- BridgeTower embeddings (from bridgetower_search)
- OpenVINO Whisper (from existing implementation)
- LanceDB vector store (from existing implementation)
- MCP FastMCP server framework
- VLM inference (from existing implementation)

## Usage Example

### 1. Start the Server
```bash
cd src
python main_realtime_capture_server.py
```

### 2. Use the Client
```bash
cd src
python mcp_clients/realtime_capture_client.py
```

### 3. Basic Workflow
1. **Start Capture**: Begin webcam and microphone recording
2. **Wait**: Let the system capture some audio-visual data (30-60 seconds)
3. **Process**: Convert recent captures into searchable embeddings
4. **Search**: Query the system about what was captured
5. **Stop**: End the capture session

### 4. Integration with Existing System
The real-time capture server can be integrated with the existing multi-agent workflow by adding it as a third MCP server and creating a corresponding agent.

## Configuration

### Environment Variables
Uses the same `.env` configuration as the main project:
```env
# BridgeTower Models (existing)
TEXT_VISION_BRIDGETOWER_MODEL_PATH=...
TEXT_BRIDGETOWER_MODEL_PATH=...

# OpenVINO Whisper (existing)  
OPENVINO_WHISPER_MODEL_DIR=...

# Vector Database (existing)
LANCEDB_HOST_FILE=...
VECTORSTORE_TBL_NAME=realtime_mmrag  # New table for real-time data
```

### Capture Parameters
- **Frame Rate**: 1-10 fps (default: 2 fps)
- **Audio Duration**: 3-10 seconds per segment (default: 5 seconds)
- **Buffer Sizes**: Configurable in code (100 frames, 20 audio segments)

## Use Cases

### 1. **Interactive Demonstrations**
- Live capture during presentations
- Real-time Q&A about visual content
- Dynamic content analysis

### 2. **Meeting Assistant**
- Capture meeting discussions and visuals
- Search through meeting content
- Visual context for audio discussions

### 3. **Educational Applications**
- Interactive learning sessions
- Visual demonstration capture
- Student interaction analysis

### 4. **Content Creation**
- Live streaming with searchable content
- Tutorial creation with indexed segments
- Interactive content development

## Limitations & Considerations

### 1. **Performance**
- Real-time processing is computationally intensive
- GPU acceleration recommended for BridgeTower and VLM
- Consider reducing frame rate for lower-end hardware

### 2. **Storage**
- Vector database grows with continuous use
- Implement data retention policies for production use
- Monitor disk space for temporary files

### 3. **Privacy**
- Audio and video data is processed locally
- Temporary files are created during processing
- Consider data encryption for sensitive applications

### 4. **Hardware Requirements**
- Webcam (USB or built-in)
- Microphone (USB or built-in)
- GPU recommended for real-time inference
- Sufficient RAM for buffer management

## Future Enhancements

1. **Motion Detection**: Only process frames with significant changes
2. **Voice Activity Detection**: Skip silent audio segments
3. **Multiple Camera Support**: Support for multiple webcam inputs
4. **Real-time Streaming**: WebRTC integration for browser access
5. **Background Processing**: Continuous processing without manual triggers
6. **Data Persistence**: Configurable long-term storage options
