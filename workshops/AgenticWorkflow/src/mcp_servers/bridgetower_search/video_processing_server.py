from mcp.server.fastmcp import FastMCP, Context
from typing import List, Tuple
import os
import json
from pathlib import Path
from .utils.utils import (
    save_video_file, video_to_audio, 
    extract_transcript_from_audio_with_openvino,
    extract_transcript_from_audio, 
    extract_and_save_frames_and_metadata,
    refine_transcript_for_ingestion_and_inference_from_metadatas,
    ingest_text_image_pairs_to_vectorstore, 
)
import asyncio
from utils.logger import log
from utils.utils import load_env

import lancedb
from mcp_servers.bridgetower_search.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mcp_servers.bridgetower_search.vectorstores.multimodal_lancedb import MultimodalLanceDB
from mcp_servers.vlm_inference.openvino_multimodal import vlm_inference
from mcp import types
import base64


mcp = FastMCP(name="VideoRetrieverServer", port=3000, host="localhost")
load_env()
openvino_whisper_model_dir = os.getenv("OPENVINO_WHISPER_MODEL_DIR")
TBL_NAME = os.getenv("VECTORSTORE_TBL_NAME", "mmrag")
LANCEDB_HOST_FILE = os.getenv("LANCEDB_HOST_FILE", "./lancedb_vectorstore/.lancedb")
_db = lancedb.connect(LANCEDB_HOST_FILE)
VECTORSTORE_INGESTION_MODE = os.getenv("VECTORSTORE_INGESTION_MODE", "overwrite")
# for bridgetower embeddings
TEXT_VISION_BRIDGETOWER_MODEL_PATH = os.getenv("TEXT_VISION_BRIDGETOWER_MODEL_PATH")
TEXT_BRIDGETOWER_MODEL_PATH = os.getenv("TEXT_BRIDGETOWER_MODEL_PATH")
VISION_BRIDGETOWER_MODEL_PATH = os.getenv("VISION_BRIDGETOWER_MODEL_PATH")
_embedder = BridgeTowerEmbeddings(
    text_vision_model_path=TEXT_VISION_BRIDGETOWER_MODEL_PATH,
    text_model_path=TEXT_BRIDGETOWER_MODEL_PATH,
    vision_model_path=VISION_BRIDGETOWER_MODEL_PATH,
)


@mcp.tool()
async def ingest_videos(b64_file: bytes, filename: str, ctx: Context) -> str:
    """
    Preprocess base64 encoded video file and ingest it to the database.

    Args:
        file (bytes): The video file content.
        filename (str): The name of the video file.

    Returns:
        str: Confirmation message.
    """
    tmp_path = os.path.join(os.getcwd(), "tmp_videos")
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    cur_process = 0
    total = 6
    
    await ctx.info(f"Saving video file {filename} to {tmp_path}")
    cur_process += 1
    await ctx.report_progress(
                progress=cur_process,
                total=total,
                message=f"Saving video...",
            )
    full_video_path = save_video_file(b64_file, filename, path_to_save=tmp_path)

    await ctx.info(f"Extracting audio from video")
    cur_process += 1
    await ctx.report_progress(
                progress=cur_process,
                total=total,
                message=f"Extracting audio from video...",
            )
    full_audio_path = video_to_audio(full_video_path, filename, path_to_save=tmp_path)

    await ctx.info(f"Extracting transcript from audio")
    cur_process += 1
    await ctx.report_progress(
                progress=cur_process,
                total=total,
                message=f"Extracting transcript from audio...",
            )
    # Use OpenVINO whisper with the proper model path
    full_transcript_path = extract_transcript_from_audio_with_openvino(full_audio_path, 
                                                                       filename, 
                                                                       path_to_save=tmp_path, 
                                                                       model_dir=openvino_whisper_model_dir)
    
    await ctx.info(f"Extracting frames and transcripts from video and audio")
    cur_process += 1
    await ctx.report_progress(
                progress=cur_process,
                total=total,
                message=f"Extracting frames and transcripts from video and audio...",
            )
    path_to_save_extracted_frames = os.path.join(tmp_path, "extracted_frames")
    Path(path_to_save_extracted_frames).mkdir(parents=True, exist_ok=True)

    metadatas = extract_and_save_frames_and_metadata(
        path_to_video=full_video_path, 
        path_to_transcript=full_transcript_path, 
        path_to_save_extracted_frames=path_to_save_extracted_frames, 
        path_to_save_metadatas=tmp_path
    )

    await ctx.info(f"Refining transcript for ingestion and inference")
    cur_process += 1
    await ctx.report_progress(
                progress=cur_process,
                total=total,
                message=f"Refining transcript for ingestion and inference...",
        )
    text_list, image_list, refined_metadatas = refine_transcript_for_ingestion_and_inference_from_metadatas(metadatas)

    await ctx.info(f"Ingesting extracted frame and refined transcript to vectorstore")
    cur_process += 1
    await ctx.report_progress(
                progress=cur_process,
                total=total,
                message=f"Ingesting extracted frame and refined transcript to vectorstore...",
        )
    
    instance = ingest_text_image_pairs_to_vectorstore(
        texts=text_list,
        images=image_list,
        embedding=_embedder,
        metadatas=refined_metadatas,
        connection=_db,
        table_name=TBL_NAME,
        mode=VECTORSTORE_INGESTION_MODE,
    )
    return f"Video saved at {full_video_path}\nAudio saved at {full_audio_path}\nTranscript saved at {full_transcript_path}"

@mcp.tool()
async def search_from_video(
    query: str,   
    ctx: Context,
    top_k: int = 1
) -> str:
    """
    Useful for answering queries that requires searching from video.

    Args:
        query (str): The search query.

    Returns:
        str: JSON string containing text response and base64-encoded image data.
    """
    await ctx.info(f"Searching from video with query: {query} and top_k: {top_k}")
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE, 
        embedding=_embedder, 
        table_name=TBL_NAME
    )

    retriever = vectorstore.as_retriever(
        search_type='similarity', 
        search_kwargs={"k": top_k}
    )
    
    results = retriever.invoke(query)
    response = []
    image_paths = []
    if results:        
        await ctx.info(f"Found {len(results)} results for the query.")
        for result in results:
            frame_path = result.metadata.get('extracted_frame_path', None)
            transcript = result.metadata.get('transcript_for_inference', None)
            if frame_path and transcript:
                with open(frame_path, "rb") as f:
                    image_bytes = base64.b64encode(f.read()).decode("utf-8")
                image_paths.append(frame_path)
                response.append((image_bytes, transcript))
    
    if not response:
        await ctx.info("No results found for the query.")
        return json.dumps({
            "_meta": {"error": "No results found for the query"},
            "content": [
                {"type": "text", "text": "No results found for the query."}
            ]
        })
    
    await ctx.info(f"VLM Inferencing...")
    final_response_text = vlm_inference(
        retrieval_messages=[(types.ImageContent(type="image", mimeType="jpeg", data=img), text) for img, text in response],
        query=query,
    )
    
    # Check if VLM response is valid
    if not isinstance(final_response_text, str):
        await ctx.info(f"VLM returned non-string response: {type(final_response_text)}")
        if hasattr(final_response_text, 'text'):
            final_response_text = final_response_text.text
        else:
            final_response_text = str(final_response_text)
    
    print(f"Final response: {final_response_text}")
    
    # Return JSON format that the callback expects
    result_data = {
        "_meta": {"query": query, "results_count": len(response)},
        "content": [
            {"type": "text", "text": final_response_text}
        ]
    }
    
    # Add image data to the response
    for image_b64, transcript in response:
        result_data["content"].append({
            "type": "image", 
            "data": image_b64,
            "transcript": transcript
        })
    
    return json.dumps(result_data)

# @mcp.tool()
# async def long_running_task(ctx: Context):
#   for i in range(10):
#     await asyncio.sleep(1) # Simulate work
#     await ctx.report_progress(
#                 progress=i + 1,
#                 total=10,
#                 message=f"Processing step {i + 1} of 10",
#             )
#     await ctx.debug(f"Debug Completed step {i + 1}")
#     await ctx.info(f"INFO Completed step {i + 1}")
#     log.info(f"Completed step {i + 1}")
#   return "Task complete!"