from typing import Iterator, TextIO
from typing import List, Optional, Any
import PIL
import textwrap
import cv2
from io import StringIO
from os import path as osp
import json
import webvtt
from pathlib import Path
import base64
from moviepy.video.io.VideoFileClip import VideoFileClip
#from .notebook_utils import device_widget

from langchain_core.embeddings import Embeddings
#from mcp_servers.bridgetower_search.vectorstores.multimodal_lancedb import MultimodalLanceDB
import openvino_genai as ov_genai
# import whisper  # Commented out due to Windows compatibility issues - using OpenVINO whisper instead


def video_to_audio(path_to_video: str, filename: str, path_to_save: str = ""):
    """
    Extract audio from video file and save it as WAV file in the required format.

    Args:
        path_to_video (str): Path to the video file.
        filename (str): Name of the video file.
        path_to_save (str): The path to save the extracted audio file.
    """
    try:
        Path(osp.abspath(path_to_save)).mkdir(parents=True, exist_ok=True)

        # Create WAV filename
        if filename.endswith('.mp4'):
            filename = filename[:-4]
        if not filename.endswith('.wav'):
            filename = filename + ".wav"

        output_audio_path = osp.join(path_to_save, filename)

        # Check if video file exists
        if not osp.exists(path_to_video):
            raise FileNotFoundError(f"Video file not found: {path_to_video}")

        # Extract audio using moviepy with timeout protection
        clip = VideoFileClip(path_to_video)

        if clip.audio is None:
            raise ValueError("No audio track found in video file")

        audio = clip.audio

        # Set timeout to prevent hanging
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Audio extraction timed out")
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

        try:
            # Write audio as WAV: IEEE Float, mono, 16000 Hz
            audio.write_audiofile(
                output_audio_path,
                fps=16000,
                nbytes=4,
                codec='pcm_f32le'
            )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        clip.close()
        return output_audio_path

    except Exception as e:
        # Try alternative method using ffmpeg
        try:
            import subprocess
            subprocess.run([
                'ffmpeg', '-i', path_to_video, '-vn',
                '-acodec', 'pcm_f32le',
                '-ar', '16000',
                '-ac', '1',
                '-y', output_audio_path
            ], check=True, capture_output=True)
            return output_audio_path
        except Exception:
            raise e


# def extract_transcript_from_audio(
#         path_to_audio: str, 
#         filename: str,
#         path_to_save: str, 
#         model_dir: str):
#     """
#     Extract transcript from audio file using OpenVINO GenAI whisper-small model
#     Args:
#         path_to_audio (str): Path to the audio file.
#         path_to_save_transcript (str): Path to save the transcript file.
#     """
#     import whisper  # Local import to avoid Windows compatibility issues at module level
#     whisper_model = whisper.load_model(model_dir)
#     # Use "transcribe" for English audio, "translate" for non-English to English
#     options = dict(task="transcribe", best_of=1, language='en')
#     results = whisper_model.transcribe(path_to_audio, **options)
#     vtt = getSubs(results["segments"], "vtt")
#     # # save the transcript to a file
#     if filename.endswith('.mp4'):
#         filename = filename[:-4]  # remove .mp4 extension
#     if not filename.endswith('.vtt'):
#         filename = filename + ".vtt"
#     output_transcript_path = osp.join(path_to_save, filename)
#     with open(output_transcript_path, 'w', encoding='utf-8') as f:
#         f.write(vtt)
#     return output_transcript_path








# #########################################OLD CODE#########################################
# def format_timestamp(seconds: float):
#     """
#     format time in srt-file expected format
#     """
#     assert seconds >= 0, "non-negative timestamp expected"
#     milliseconds = round(seconds * 1000.0)

#     hours = milliseconds // 3_600_000
#     milliseconds -= hours * 3_600_000

#     minutes = milliseconds // 60_000
#     milliseconds -= minutes * 60_000

#     seconds = milliseconds // 1_000
#     milliseconds -= seconds * 1_000

#     return (f"{hours}:" if hours > 0 else "00:") + f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# # def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
# #     assert seconds >= 0, "non-negative timestamp expected"
# #     milliseconds = round(seconds * 1000.0)

# #     hours = milliseconds // 3_600_000
# #     milliseconds -= hours * 3_600_000

# #     minutes = milliseconds // 60_000
# #     milliseconds -= minutes * 60_000

# #     seconds = milliseconds // 1_000
# #     milliseconds -= seconds * 1_000

# #     hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
# #     return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

# # a help function that helps to convert a specific time written as a string in format `webvtt` into a time in miliseconds
# def str2time(strtime):
#     # strip character " if exists
#     strtime = strtime.strip('"')
#     # get hour, minute, second from time string
#     hrs, mins, seconds = [float(c) for c in strtime.split(':')]
#     # get the corresponding time as total seconds 
#     total_seconds = hrs * 60**2 + mins * 60 + seconds
#     total_miliseconds = total_seconds * 1000
#     return total_miliseconds


# def _processText(text: str, maxLineWidth=None):
#     if (maxLineWidth is None or maxLineWidth < 0):
#         return text

#     lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
#     return '\n'.join(lines)


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
         # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
         # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

#     # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)