from mcp.server.fastmcp import FastMCP, Context
#import json
import os
import sys
import yaml
from pathlib import Path
from openai import OpenAI
import base64
import lancedb
import os.path as osp
#from mcp import types

from utils import video_to_audio
#   extract_transcript_from_audio_with_openvino,
#   extract_transcript_from_audio,
#   extract_and_save_frames_and_metadata,
#   refine_transcript_for_ingestion_and_inference_from_metadatas,
#   ingest_text_image_pairs_to_vectorstore,

#from bridgetower_embeddings import BridgeTowerEmbeddings
#from multimodal_lancedb import MultimodalLanceDB
#from openvino_multimodal import vlm_inference


class VideoRetrieverServer:
    """Video retrieval server with MCP integration."""

    def __init__(self, config_path: str):
        """Initialize the VideoRetrieverServer with configuration.

        Args:
            config_path: Path to YAML configuration file. File must exist.

        Raises:
            ValueError: If config_path is None or empty.
            SystemExit: If configuration file doesn't exist or can't be loaded.
        """
        if not config_path or not config_path.strip():
            raise ValueError("config_path is required and cannot be empty")

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._initialize_resources()

        # Initialize MCP server once
        self.mcp_server = FastMCP(
            name="VideoRetrieverServer",
            port=3000,
            host="0.0.0.0",
        )

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(
                "Error: Configuration file "
                f"'{self.config_path}' not found. Exiting."
            )
            sys.exit(1)

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config is not None else {}
        except Exception as e:
            print(
                "Error: Failed to load configuration file "
                f"'{self.config_path}': {e}"
            )
            sys.exit(1)

    def _get_config_value(self, path: str, default=None):
        """Get value from config with fallback to default."""
        try:
            # Navigate through nested config path
            # (e.g., 'openvino_whisper.model_dir')
            keys = path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def _validate_config(self):
        """Validate that all required configuration values are present."""
        # Validate audio transcription model directory
        self.audio_transcription_model_dir = self._get_config_value(
            'audio_transcription.model_dir', None
        )
        if not self.audio_transcription_model_dir:
            raise ValueError(
                "Missing required audio transcription model directory "
                "in configuration"
            )

        # Validate BridgeTower model paths
        self.text_vision_model_path = self._get_config_value(
            'bridgetower.text_vision_model_path', None
        )
        self.text_model_path = self._get_config_value(
            'bridgetower.text_model_path', None
        )
        self.vision_model_path = self._get_config_value(
            'bridgetower.vision_model_path', None
        )

        missing_paths = []
        if not self.text_vision_model_path:
            missing_paths.append('text_vision_model_path')
        if not self.text_model_path:
            missing_paths.append('text_model_path')
        if not self.vision_model_path:
            missing_paths.append('vision_model_path')

        if missing_paths:
            raise ValueError(
                "Missing required BridgeTower model paths in configuration: "
                f"{', '.join(missing_paths)}"
            )

        # Set other configuration values
        self.table_name = self._get_config_value(
            'bridgetower.vectorstore.table_name', 'mmrag'
        )
        self.lancedb_host_file = self._get_config_value(
            'bridgetower.vectorstore.lancedb_host_file',
            './lancedb_vectorstore/.lancedb'
        )
        self.ingestion_mode = self._get_config_value(
            'bridgetower.vectorstore.ingestion_mode', 'overwrite'
        )

    def _initialize_resources(self):
        """Initialize external resources like DB connections and models."""
        # Connect to LanceDB
        self.db = lancedb.connect(self.lancedb_host_file)

        # Initialize BridgeTower embedder
        # self.embedder = BridgeTowerEmbeddings(
        #     text_vision_model_path=self.text_vision_model_path,
        #     text_model_path=self.text_model_path,
        #     vision_model_path=self.vision_model_path,
        # )

    async def ingest_videos(
        self, b64_file, filename: str, ctx: Context
    ) -> str:
        """Preprocess video file (base64 or file path) and ingest it to database."""
        tmp_path = os.path.join(os.getcwd(), "tmp_videos")

        # Ensure tmp_videos directory exists
        os.makedirs(tmp_path, exist_ok=True)

        # Handle both file path and base64 encoded content
        video_file_path = osp.join(tmp_path, filename)

        if isinstance(b64_file, str):
            # Check if it's a file path
            if osp.exists(b64_file) and b64_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                import shutil
                shutil.copy2(b64_file, video_file_path)
            else:
                # Assume it's base64 encoded content
                with open(video_file_path, "wb") as f:
                    f.write(base64.b64decode(b64_file))
        elif isinstance(b64_file, bytes):
            # Direct base64 bytes
            with open(video_file_path, "wb") as f:
                f.write(base64.b64decode(b64_file))
        else:
            raise ValueError(f"Unsupported b64_file type: {type(b64_file)}")

        # Extract audio from the saved video file
        audio_path = video_to_audio(video_file_path, filename, path_to_save=tmp_path)

        # Extract transcript from audio using OpenVINO whisper
        client = OpenAI(api_key="unset", base_url="http://localhost:8000/v3")

        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper",
                    file=audio_file
                )
        except Exception:
            # Create a dummy transcript to prevent process failure
            transcript = type('Transcript', (), {'text': 'Transcription failed - audio processing error'})()

        # Save transcript to file for further processing
        # Handle both .mp4 and .wav extensions
        base_filename = filename
        if base_filename.endswith('.mp4'):
            base_filename = base_filename[:-4]  # remove .mp4 extension
        elif base_filename.endswith('.wav'):
            base_filename = base_filename[:-4]  # remove .wav extension

        transcript_filename = f"{base_filename}_transcript.txt"
        full_transcript_path = os.path.join(tmp_path, transcript_filename)
        with open(full_transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript.text)

        #### PERFECT UNTIL HERE ####

        path_to_save_extracted_frames = os.path.join(
            tmp_path, "extracted_frames"
        )
        #########

        #########
        
        metadatas = extract_and_save_frames_and_metadata(
            path_to_video=osp.join(tmp_path, filename),
            path_to_transcript=full_transcript_path,
            path_to_save_extracted_frames=path_to_save_extracted_frames,
            path_to_save_metadatas=tmp_path,
        )

        (
            text_list,
            image_list,
            refined_metadatas
        ) = refine_transcript_for_ingestion_and_inference_from_metadatas(
            metadatas
        )

        ingest_text_image_pairs_to_vectorstore(
            texts=text_list,
            images=image_list,
            embedding=self.embedder,
            metadatas=refined_metadatas,
            connection=self.db,
            table_name=self.table_name,
            mode=self.ingestion_mode,
        )

        await ctx.info(f"Video processing complete for {filename}")
        return (
            f"Video saved at {osp.join(tmp_path, filename)}\n"
            f"Audio saved at {full_audio_path}\n"
            f"Transcript saved at {full_transcript_path}"
        )

    def get_mcp_server(self):
        """Get the MCP server instance for this video retriever."""
        return self.mcp_server

    def run(self):
        """Run the MCP server."""
        # Use SSE (Server-Sent Events) for streaming
        import asyncio
        asyncio.run(self.mcp_server.run_sse_async())


if __name__ == "__main__":
    # Initialize server and register tools
    server = VideoRetrieverServer("NEW_mcp_info.yaml")

    # Register the ingest_videos tool
    @server.mcp_server.tool()
    async def ingest_videos(b64_file, filename: str, ctx):
        """Preprocess video file (base64 or file path) and ingest it to database."""
        return await server.ingest_videos(b64_file, filename, ctx)

    # Start the server
    server.run()