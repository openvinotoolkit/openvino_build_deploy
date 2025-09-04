"""
Real-time Webcam and Microphone Capture Server

This MCP server captures live video frames from a webcam and audio from a microphone,
processes them using BridgeTower embeddings, and stores them in a vector database
for real-time multimodal search and query capabilities.

Key Features:
- Live webcam frame capture
- Real-time microphone audio recording and transcription
- BridgeTower multimodal embedding generation
- Vector database storage and similarity search
- Streaming MCP server interface
"""

from mcp.server.fastmcp import FastMCP, Context
from typing import List, Dict, Optional, Tuple
import asyncio
import cv2
import numpy as np
import sounddevice as sd
import wave
import tempfile
import os
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import time
from collections import deque

# Import existing components from the project
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import log
from utils.utils import load_env
import lancedb
from mcp_servers.bridgetower_search.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mcp_servers.bridgetower_search.vectorstores.multimodal_lancedb import MultimodalLanceDB
from mcp_servers.vlm_inference.openvino_multimodal import vlm_inference
from mcp_servers.bridgetower_search.utils.utils import extract_transcript_from_audio_with_openvino
from mcp import types


class RealtimeCaptureManager:
    """Manages real-time capture from webcam and microphone"""
    
    def __init__(self, frame_rate: int = 2, audio_duration: int = 5):
        self.frame_rate = frame_rate  # Frames per second to capture
        self.audio_duration = audio_duration  # Seconds of audio to capture at a time
        self.sample_rate = 16000  # Audio sample rate for Whisper
        
        # Threading controls
        self.capture_active = False
        self.frame_thread = None
        self.audio_thread = None
        
        # Data queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=5)
        
        # Storage for captured data
        self.frame_buffer = deque(maxlen=100)  # Keep last 100 frames
        self.audio_buffer = deque(maxlen=20)   # Keep last 20 audio segments
        
        # Camera and audio setup
        self.camera = None
        self.temp_dir = tempfile.mkdtemp()
        
    def start_capture(self):
        """Start capturing from webcam and microphone"""
        if self.capture_active:
            return "Capture already active"
            
        self.capture_active = True
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)  # Use default webcam
        if not self.camera.isOpened():
            raise RuntimeError("Cannot open webcam")
            
        # Set camera properties for better quality
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Start capture threads
        self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.audio_thread = threading.Thread(target=self._capture_audio, daemon=True)
        
        self.frame_thread.start()
        self.audio_thread.start()
        
        log.info("Started real-time capture from webcam and microphone")
        return "Real-time capture started successfully"
    
    def stop_capture(self):
        """Stop capturing from webcam and microphone"""
        if not self.capture_active:
            return "Capture not active"
            
        self.capture_active = False
        
        # Wait for threads to finish
        if self.frame_thread:
            self.frame_thread.join(timeout=2.0)
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
            
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
            
        log.info("Stopped real-time capture")
        return "Real-time capture stopped successfully"
    
    def _capture_frames(self):
        """Background thread for capturing video frames"""
        frame_interval = 1.0 / self.frame_rate
        last_capture_time = 0
        
        while self.capture_active:
            current_time = time.time()
            
            if current_time - last_capture_time >= frame_interval:
                ret, frame = self.camera.read()
                if ret:
                    # Convert frame to base64
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Create frame metadata
                    frame_data = {
                        'timestamp': datetime.now().isoformat(),
                        'frame_base64': frame_b64,
                        'frame_id': f"frame_{int(current_time * 1000)}"
                    }
                    
                    # Add to buffer and queue
                    self.frame_buffer.append(frame_data)
                    
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Full:
                        # Remove oldest frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame_data)
                        except queue.Empty:
                            pass
                    
                    last_capture_time = current_time
                else:
                    log.warning("Failed to capture frame from webcam")
                    time.sleep(0.1)
            else:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def _capture_audio(self):
        """Background thread for capturing audio segments"""
        while self.capture_active:
            try:
                # Record audio segment
                log.info(f"Recording {self.audio_duration} seconds of audio...")
                audio_data = sd.rec(
                    int(self.sample_rate * self.audio_duration),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                sd.wait()  # Wait until recording is finished
                
                # Save to temporary wav file
                timestamp = datetime.now().isoformat().replace(':', '-')
                temp_audio_path = os.path.join(self.temp_dir, f"audio_{timestamp}.wav")
                
                # Convert to int16 and save
                audio_int16 = (audio_data * 32767).astype(np.int16)
                with wave.open(temp_audio_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Create audio metadata
                audio_data = {
                    'timestamp': datetime.now().isoformat(),
                    'audio_path': temp_audio_path,
                    'duration': self.audio_duration,
                    'audio_id': f"audio_{int(time.time() * 1000)}"
                }
                
                # Add to buffer and queue
                self.audio_buffer.append(audio_data)
                
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    # Remove oldest audio if queue is full
                    try:
                        old_audio = self.audio_queue.get_nowait()
                        # Clean up old audio file
                        if os.path.exists(old_audio['audio_path']):
                            os.remove(old_audio['audio_path'])
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                log.error(f"Error capturing audio: {e}")
                time.sleep(1.0)  # Wait before retrying
    
    def get_recent_frames(self, count: int = 5) -> List[Dict]:
        """Get the most recent captured frames"""
        return list(self.frame_buffer)[-count:] if len(self.frame_buffer) >= count else list(self.frame_buffer)
    
    def get_recent_audio(self, count: int = 2) -> List[Dict]:
        """Get the most recent captured audio segments"""
        return list(self.audio_buffer)[-count:] if len(self.audio_buffer) >= count else list(self.audio_buffer)


# Initialize MCP server
mcp = FastMCP(name="RealtimeCaptureServer", port=3002, host="localhost")

# Load environment and initialize components
load_env()
openvino_whisper_model_dir = os.getenv("OPENVINO_WHISPER_MODEL_DIR")
TBL_NAME = os.getenv("VECTORSTORE_TBL_NAME", "realtime_mmrag")
LANCEDB_HOST_FILE = os.getenv("LANCEDB_HOST_FILE", "./lancedb_vectorstore/.lancedb")
_db = lancedb.connect(LANCEDB_HOST_FILE)

# Initialize BridgeTower embeddings
TEXT_VISION_BRIDGETOWER_MODEL_PATH = os.getenv("TEXT_VISION_BRIDGETOWER_MODEL_PATH")
TEXT_BRIDGETOWER_MODEL_PATH = os.getenv("TEXT_BRIDGETOWER_MODEL_PATH")
VISION_BRIDGETOWER_MODEL_PATH = os.getenv("VISION_BRIDGETOWER_MODEL_PATH")
_embedder = BridgeTowerEmbeddings(
    text_vision_model_path=TEXT_VISION_BRIDGETOWER_MODEL_PATH,
    text_model_path=TEXT_BRIDGETOWER_MODEL_PATH,
    vision_model_path=VISION_BRIDGETOWER_MODEL_PATH,
)

# Global capture manager
_capture_manager = RealtimeCaptureManager()


@mcp.tool()
async def start_realtime_capture(frame_rate: int = 2, audio_duration: int = 5, ctx: Context = None) -> str:
    """
    Start real-time capture from webcam and microphone.
    
    Args:
        frame_rate (int): Number of frames per second to capture from webcam
        audio_duration (int): Duration in seconds for each audio segment
        
    Returns:
        str: Status message
    """
    try:
        global _capture_manager
        _capture_manager = RealtimeCaptureManager(frame_rate, audio_duration)
        result = _capture_manager.start_capture()
        
        if ctx:
            await ctx.info(f"Started real-time capture with {frame_rate} fps and {audio_duration}s audio segments")
        
        log.info(f"Real-time capture started: {frame_rate} fps, {audio_duration}s audio")
        return result
        
    except Exception as e:
        error_msg = f"Failed to start real-time capture: {str(e)}"
        log.error(error_msg)
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def stop_realtime_capture(ctx: Context = None) -> str:
    """
    Stop real-time capture from webcam and microphone.
    
    Returns:
        str: Status message
    """
    try:
        result = _capture_manager.stop_capture()
        
        if ctx:
            await ctx.info("Stopped real-time capture")
        
        log.info("Real-time capture stopped")
        return result
        
    except Exception as e:
        error_msg = f"Failed to stop real-time capture: {str(e)}"
        log.error(error_msg)
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def process_and_store_recent_data(ctx: Context = None, frame_count: int = 3, audio_count: int = 1) -> str:
    """
    Process recent captured frames and audio, generate embeddings, and store in vector database.
    
    Args:
        frame_count (int): Number of recent frames to process
        audio_count (int): Number of recent audio segments to process
        
    Returns:
        str: Processing status and results
    """
    try:
        if not _capture_manager.capture_active:
            return "Real-time capture is not active. Please start capture first."
        
        # Get recent frames and audio
        recent_frames = _capture_manager.get_recent_frames(frame_count)
        recent_audio = _capture_manager.get_recent_audio(audio_count)
        
        if not recent_frames:
            return "No frames available for processing"
        
        if not recent_audio:
            return "No audio available for processing"
        
        if ctx:
            await ctx.info(f"Processing {len(recent_frames)} frames and {len(recent_audio)} audio segments")
        
        processed_pairs = []
        
        # Process each audio segment
        for audio_data in recent_audio:
            audio_path = audio_data['audio_path']
            
            if ctx:
                await ctx.info(f"Transcribing audio segment: {audio_data['audio_id']}")
            
            # Extract transcript using OpenVINO Whisper
            try:
                transcript_path = extract_transcript_from_audio_with_openvino(
                    path_to_audio=audio_path,
                    filename=audio_data['audio_id'],
                    path_to_save=_capture_manager.temp_dir,
                    model_dir=openvino_whisper_model_dir
                )
                
                # Read transcript
                transcript_text = ""
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        import webvtt
                        for caption in webvtt.read(transcript_path):
                            transcript_text += caption.text + " "
                
                if not transcript_text.strip():
                    transcript_text = "No speech detected in this audio segment"
                
            except Exception as e:
                log.warning(f"Failed to transcribe audio {audio_data['audio_id']}: {e}")
                transcript_text = "Audio transcription failed"
            
            # Find frames that correspond to this audio segment (by timestamp proximity)
            audio_time = datetime.fromisoformat(audio_data['timestamp'])
            corresponding_frames = []
            
            for frame_data in recent_frames:
                frame_time = datetime.fromisoformat(frame_data['timestamp'])
                time_diff = abs((frame_time - audio_time).total_seconds())
                
                # Include frames within the audio duration window
                if time_diff <= audio_data['duration'] + 2:  # +2 seconds buffer
                    corresponding_frames.append(frame_data)
            
            # If no corresponding frames, use the most recent frame
            if not corresponding_frames and recent_frames:
                corresponding_frames = [recent_frames[-1]]
            
            # Create text-image pairs for each corresponding frame
            for frame_data in corresponding_frames:
                processed_pairs.append({
                    'text': transcript_text.strip(),
                    'image_base64': frame_data['frame_base64'],
                    'timestamp': frame_data['timestamp'],
                    'audio_id': audio_data['audio_id'],
                    'frame_id': frame_data['frame_id'],
                    'source': 'realtime_capture'
                })
        
        if not processed_pairs:
            return "No valid text-image pairs could be created"
        
        if ctx:
            await ctx.info(f"Generated {len(processed_pairs)} text-image pairs, storing in vector database")
        
        # Prepare data for vector store
        texts = [pair['text'] for pair in processed_pairs]
        images = []
        metadatas = []
        
        for pair in processed_pairs:
            # Decode base64 image
            image_bytes = base64.b64decode(pair['image_base64'])
            temp_image_path = os.path.join(_capture_manager.temp_dir, f"{pair['frame_id']}.jpg")
            with open(temp_image_path, 'wb') as f:
                f.write(image_bytes)
            
            images.append(temp_image_path)
            metadatas.append({
                'timestamp': pair['timestamp'],
                'audio_id': pair['audio_id'],
                'frame_id': pair['frame_id'],
                'source': pair['source'],
                'image_base64': pair['image_base64'],  # Store for later retrieval
                'transcript': pair['text']
            })
        
        # Store in vector database
        vectorstore = MultimodalLanceDB(
            uri=LANCEDB_HOST_FILE,
            embedding=_embedder,
            table_name=TBL_NAME
        )
        
        # Add documents to vector store
        from langchain.schema import Document
        documents = []
        for i, (text, image_path, metadata) in enumerate(zip(texts, images, metadatas)):
            doc = Document(
                page_content=text,
                metadata=metadata
            )
            documents.append(doc)
        
        vectorstore.add_documents(documents)
        
        # Clean up temporary image files
        for image_path in images:
            if os.path.exists(image_path):
                os.remove(image_path)
        
        result_msg = f"Successfully processed and stored {len(processed_pairs)} text-image pairs from real-time capture"
        log.info(result_msg)
        
        if ctx:
            await ctx.info(result_msg)
        
        return result_msg
        
    except Exception as e:
        error_msg = f"Failed to process and store recent data: {str(e)}"
        log.error(error_msg)
        if ctx:
            await ctx.error(error_msg)
        return error_msg


@mcp.tool()
async def search_realtime_data(query: str, top_k: int = 3, ctx: Context = None) -> str:
    """
    Search through stored real-time capture data using text queries.
    
    Args:
        query (str): Search query
        top_k (int): Number of top results to return
        
    Returns:
        str: JSON string containing search results with text and images
    """
    try:
        if ctx:
            await ctx.info(f"Searching real-time data with query: '{query}'")
        
        # Initialize vector store
        vectorstore = MultimodalLanceDB(
            uri=LANCEDB_HOST_FILE,
            embedding=_embedder,
            table_name=TBL_NAME
        )
        
        # Perform similarity search
        retriever = vectorstore.as_retriever(
            search_type='similarity',
            search_kwargs={"k": top_k}
        )
        
        results = retriever.invoke(query)
        
        if not results:
            return json.dumps({
                "_meta": {"query": query, "results_count": 0},
                "content": [{"type": "text", "text": "No matching results found in real-time capture data."}]
            })
        
        # Prepare response with VLM analysis
        response_pairs = []
        for result in results:
            metadata = result.metadata
            if metadata.get('image_base64') and metadata.get('transcript'):
                response_pairs.append((
                    metadata['image_base64'],
                    metadata['transcript']
                ))
        
        # Get VLM analysis of the results
        if response_pairs:
            if ctx:
                await ctx.info("Analyzing results with VLM...")
            
            vlm_response = vlm_inference(
                retrieval_messages=[(
                    types.ImageContent(type="image", mimeType="jpeg", data=img),
                    text
                ) for img, text in response_pairs],
                query=query,
            )
            
            if not isinstance(vlm_response, str):
                if hasattr(vlm_response, 'text'):
                    vlm_response = vlm_response.text
                else:
                    vlm_response = str(vlm_response)
        else:
            vlm_response = "Found relevant results but no images available for analysis."
        
        # Build JSON response
        response_data = {
            "_meta": {
                "query": query,
                "results_count": len(results),
                "timestamp": datetime.now().isoformat(),
                "source": "realtime_capture"
            },
            "content": [
                {"type": "text", "text": vlm_response}
            ]
        }
        
        # Add image results
        for result in results:
            metadata = result.metadata
            if metadata.get('image_base64'):
                response_data["content"].append({
                    "type": "image",
                    "data": metadata['image_base64'],
                    "timestamp": metadata.get('timestamp', ''),
                    "frame_id": metadata.get('frame_id', ''),
                    "transcript": metadata.get('transcript', '')
                })
        
        log.info(f"Real-time search completed: {len(results)} results for query '{query}'")
        
        if ctx:
            await ctx.info(f"Found {len(results)} results, returning response with VLM analysis")
        
        return json.dumps(response_data)
        
    except Exception as e:
        error_msg = f"Failed to search real-time data: {str(e)}"
        log.error(error_msg)
        if ctx:
            await ctx.error(error_msg)
        return json.dumps({
            "_meta": {"error": error_msg},
            "content": [{"type": "text", "text": f"Search failed: {error_msg}"}]
        })


@mcp.tool()
async def get_capture_status(ctx: Context = None) -> str:
    """
    Get the current status of real-time capture system.
    
    Returns:
        str: JSON string with capture status information
    """
    try:
        status_data = {
            "capture_active": _capture_manager.capture_active,
            "frame_buffer_size": len(_capture_manager.frame_buffer),
            "audio_buffer_size": len(_capture_manager.audio_buffer),
            "frame_queue_size": _capture_manager.frame_queue.qsize(),
            "audio_queue_size": _capture_manager.audio_queue.qsize(),
            "timestamp": datetime.now().isoformat()
        }
        
        if _capture_manager.capture_active:
            status_data["camera_available"] = _capture_manager.camera is not None and _capture_manager.camera.isOpened()
        
        return json.dumps(status_data, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to get capture status: {str(e)}"
        log.error(error_msg)
        return json.dumps({"error": error_msg})


if __name__ == "__main__":
    # For testing - run the server
    log.info("Starting Real-time Capture MCP Server on port 3002")
    mcp.run(transport="sse")
