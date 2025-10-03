# Copyright 2018-2021 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import imghdr
import io
import mimetypes
from typing import cast
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageFile
import hashlib
import base64

# Maximum content width
MAXIMUM_CONTENT_WIDTH = 1460  # 2 * 730

def _image_has_alpha_channel(image):
    """Check if the image has an alpha channel."""
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        return True
    else:
        return False

def _format_from_image_type(image, output_format):
    """Determine the output format based on image type."""
    output_format = output_format.upper()
    if output_format == "JPEG" or output_format == "PNG":
        return output_format

    # We are forgiving on the spelling of JPEG
    if output_format == "JPG":
        return "JPEG"

    if _image_has_alpha_channel(image):
        return "PNG"

    return "JPEG"

def _PIL_to_bytes(image, format="JPEG", quality=100):
    """Convert PIL image to bytes."""
    tmp = io.BytesIO()

    # User must have specified JPEG, so we must convert it
    if format == "JPEG" and _image_has_alpha_channel(image):
        image = image.convert("RGB")

    image.save(tmp, format=format, quality=quality)

    return tmp.getvalue()

def _BytesIO_to_bytes(data):
    """Convert BytesIO to bytes."""
    data.seek(0)
    return data.getvalue()

def _normalize_to_bytes(data, width, output_format):
    """Normalize image data to bytes with proper format and size."""
    image = Image.open(io.BytesIO(data))
    actual_width, actual_height = image.size
    format = _format_from_image_type(image, output_format)
    if output_format.lower() == "auto":
        ext = imghdr.what(None, data)
        mimetype = mimetypes.guess_type("image.%s" % ext)[0]
    else:
        mimetype = "image/" + format.lower()

    if width < 0 and actual_width > MAXIMUM_CONTENT_WIDTH:
        width = MAXIMUM_CONTENT_WIDTH

    if width > 0 and actual_width > width:
        new_height = int(1.0 * actual_height * width / actual_width)
        image = image.resize((width, new_height), resample=Image.BILINEAR)
        data = _PIL_to_bytes(image, format=format, quality=90)
        mimetype = "image/" + format.lower()

    return data, mimetype

def generate_image_hash(image, mimetype):
    """Generate a SHA-224 hash for an image."""
    hasher = hashlib.sha224()
    hasher.update(image)
    hasher.update(mimetype.encode())
    return hasher.hexdigest()

def image_to_url(image, width=-1, output_format="auto"):
    """
    Convert an image to a data URL.
    
    Args:
        image: The image data
        width: Target width (negative means preserve original if under max width)
        output_format: Output format (auto, jpeg, png)
        
    Returns:
        Data URL of the image
    """
    image_data, mimetype = _normalize_to_bytes(image, width, output_format)
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mimetype};base64,{image_base64}"

def video_to_url(video_data):
    """
    Convert video data to a data URL.
    
    Args:
        video_data: The video data
        
    Returns:
        Data URL of the video
    """
    video_base64 = base64.b64encode(video_data).decode("utf-8")
    return f"data:video/mp4;base64,{video_base64}"
