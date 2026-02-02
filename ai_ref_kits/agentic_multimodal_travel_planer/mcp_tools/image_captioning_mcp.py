"""Image captioning MCP server for the agentic multimodal travel planner.

This module provides an MCP server that uses OpenVINO Model Server (OVMS)
to perform image captioning tasks using vision-language models.
"""

import asyncio
import base64
import sys
from pathlib import Path

import yaml
from mcp.server.fastmcp import Context, FastMCP
from openai import OpenAI


class ImageCaptionServer:
    """Image captioning MCP server using OpenVINO Model Server (OVMS).

    This server provides image captioning capabilities through an MCP
    interface, using OVMS with vision-language models accessible via
    the OpenAI API format.

    Attributes:
        config_path: Path to the YAML configuration file.
        config: Loaded configuration dictionary.
        model: Model identifier from config.
        base_url: OVMS base URL.
        mcp_port: Port for the MCP server.
        client: OpenAI client instance.
        mcp_server: FastMCP server instance.
    """

    def __init__(self, config_path: str):
        """Initialize the ImageCaptionServer.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            ValueError: If config_path is empty or missing required fields.
        """
        if not config_path or not config_path.strip():
            raise ValueError("config_path is required and cannot be empty")

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._initialize_resources()

        # Initialize OpenAI client with OVMS base URL
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="dummy"
        )

        # Initialize MCP server
        self.mcp_server = FastMCP(
            name="ImageCaptionMCP",
            port=self.mcp_port,
            host="0.0.0.0",
        )

    def _load_config(self) -> dict:
        """Load YAML configuration from file.

        Returns:
            Configuration dictionary.

        Exits:
            If config file is not found or cannot be loaded.
        """
        if not self.config_path.exists():
            print(
                f"Error: Config file '{self.config_path}' not found. Exiting."
            )
            sys.exit(1)

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config is not None else {}
        except Exception as e:
            print(f"Error: Failed to load config '{self.config_path}': {e}")
            sys.exit(1)

    def _get_config_value(self, path: str, default=None):
        """Get nested config value using dotted path notation.

        Args:
            path: Dotted path to the config value (e.g., 'image_mcp.model_id').
            default: Default value if path not found.

        Returns:
            Configuration value or default if not found.
        """
        try:
            value = self.config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def _validate_config(self):
        """Validate and extract required configuration fields.

        Raises:
            ValueError: If required fields are missing from config.
        """
        self.model = self._get_config_value('image_mcp.model_id', None)
        if not self.model:
            raise ValueError("Missing required field: image_mcp.model_id")

        self.base_url = self._get_config_value(
            'image_mcp.ovms_base_url',
            None
        )
        if not self.base_url:
            raise ValueError(
                "Missing required field: image_mcp.ovms_base_url"
            )

        self.mcp_port = self._get_config_value('image_mcp.mcp_port', 3005)

    def _initialize_resources(self):
        """Initialize additional resources like headers."""
        self.headers = {"Content-Type": "application/json"}

    def encode_image(self, image_path: str) -> str:
        """Encode local image file to base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64-encoded string representation of the image.
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def image_captioning(self, image_path: str, prompt: str) -> str:
        """Generate image caption using OVMS via OpenAI API format.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt to guide the image captioning.

        Returns:
            Generated caption text for the image.
        """
        img_data = self.encode_image(image_path)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    }
                ]
            }]
        )
        return resp.choices[0].message.content

    def run(self):
        """Start the MCP server with registered tools."""
        asyncio.run(self.mcp_server.run_sse_async())


def main():
    """Main entry point for the image captioning MCP server."""
    # Initialize server with absolute config path so it works from any CWD
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "config" / "mcp_config.yaml"
    server = ImageCaptionServer(str(config_path))

    # Register the image captioning tool
    @server.mcp_server.tool()
    async def image_captioning(
        image_path: str,
        prompt: str,
        ctx: Context,
    ) -> str:
        """Generate caption for an image using vision-language model.

        Args:
            image_path: Path to the image file to caption.
            prompt: Text prompt to guide the captioning.
            ctx: MCP context object.

        Returns:
            Generated caption text.
        """
        return server.image_captioning(image_path, prompt)

    # Start the server
    server.run()


if __name__ == "__main__":
    main()
