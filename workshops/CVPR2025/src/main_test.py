from mcp_servers.bridgetower_search.openvino_bridgetower.openvino_bridgetower import OpenVINOBridgeTower
import requests
from PIL import Image

# bt = OpenVINOBridgeTower(
#     text_vision_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_large_itc.xml",
#     text_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_text_large_itc.xml",
#     vision_model_path="TBD",
# )

# embed = bt.embed(text="This is a test text for embedding.")
# print(embed)
# print(type(embed))

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "An image of two cats chilling on a couch."

# img_text_emb = bt.embed(text=text, image=image)
# print(img_text_emb)
# print(type(img_text_emb))

# from mcp_servers.bridgetower_search.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
# bte = BridgeTowerEmbeddings(
#     text_vision_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_large_itc.xml",
#     text_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_text_large_itc.xml",
#     vision_model_path="TBD",
# )

# texts = ["This is a test text for embedding."]
# embed = bte.embed_documents(texts)
# print(embed) 
# print(type(embed))
# print(embed[0])

# texts = ["This is a test text for embedding.", "Another text for testing."]
# embed = bte.embed_documents(texts)
# print(embed) 
# print(type(embed))
# print(embed[0])

# texts = ["This is a test text for embedding.", "Another text for testing.", "A third text for embedding."]
# embed = bte.embed_documents(texts)
# print(embed) 
# print(type(embed))
# print(embed[0])

# texts = ["An image of two cats chilling on a couch.", "A picture of a dog playing with a ball."]
# embed = bte.embed_image_text_pairs(
#     texts=texts,
#     images=[image, image]
# )
# print(embed)
# print(type(embed))
# print(embed[0])

import lancedb
from mcp_servers.bridgetower_search.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mcp_servers.bridgetower_search.vectorstores.multimodal_lancedb import MultimodalLanceDB
LANCEDB_HOST_FILE = "./lancedb_vectorstore/.lancedb"
TBL_NAME = "mmrag"
db = lancedb.connect(LANCEDB_HOST_FILE)

embedder = BridgeTowerEmbeddings(
    text_vision_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_large_itc.xml",
    text_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_text_large_itc.xml",
    vision_model_path="TBD",
)

# a = embedder.embed_query("This is a test text for embedding.")
# print(len(a))
# print(a)
from mcp_servers.bridgetower_search.utils.utils import (
    save_video_file, video_to_audio, 
    extract_transcript_from_audio, load_env,
    extract_and_save_frames_and_metadata,
    refine_transcript_for_ingestion_and_inference_from_metadatas,
    ingest_text_image_pairs_to_vectorstore, 
)
from pathlib import Path
import os

full_video_path = "/home/tiep/projects/demo-cvpr-tutorial/data/input_vid.mp4"
filename="test_vid1"
tmp_path = "/home/tiep/projects/demo-cvpr-tutorial/src/tmp_videos/"
print("Extracting audio from video")
full_audio_path = video_to_audio(full_video_path, filename, path_to_save=tmp_path)
print("Extracting transcript from audio...")
full_transcript_path = extract_transcript_from_audio(full_audio_path, 
                                                         filename, 
                                                         path_to_save=tmp_path, 
                                                         model_dir="/home/tiep/projects/optimum_cli_models/whisper-small")

print("extracting frames and metadata...")
path_to_save_extracted_frames = os.path.join(tmp_path, "extracted_frames")
Path(path_to_save_extracted_frames).mkdir(parents=True, exist_ok=True)
metadatas = extract_and_save_frames_and_metadata(
        path_to_video=full_video_path, 
        path_to_transcript=full_transcript_path, 
        path_to_save_extracted_frames=path_to_save_extracted_frames, 
        path_to_save_metadatas=tmp_path
)
print("refining...")
text_list, image_list, refined_metadatas = refine_transcript_for_ingestion_and_inference_from_metadatas(metadatas)
print("ingesting...")
instance = ingest_text_image_pairs_to_vectorstore(
        texts=text_list,
        images=image_list,
        embedding=embedder,
        metadatas=refined_metadatas,
        connection=db,
        table_name=TBL_NAME,
        mode="overwrite",
    )

text1 = "An image of two cats."
text2 = "There are dogs barking nearby."

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

url1='http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg'
cap1='A motorcycle sits parked across from a herd of livestock'
image2 = Image.open(requests.get(url1, stream=True).raw)
text3 = "I hear the sound of dogs barking."

texts = [text1, text2, text3]
images = [image, image, image2]
metadatas = [{
    "img_path": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "caption": texts[0]
}, {
    "img_path": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "caption": texts[1]
}, {
    "img_path": "http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg",
    "caption": texts[2]
}]
_ = MultimodalLanceDB.from_text_image_pairs(
    texts=texts,
    images=images,
    embedding=embedder,
    metadatas=metadatas,
    connection=db,
    table_name=TBL_NAME,
    mode="overwrite",
)

tbl = db.open_table(TBL_NAME)
print(f"There are {tbl.to_pandas().shape[0]} rows in the table")
pd = tbl.to_pandas()
print(pd.head(3))
print(tbl.to_pandas().columns)
# tbl.to_pandas()[['text', 'metadata']].head(3)
# print(tbl.to_pandas().iloc[0]['metadata'])
# print(tbl.to_pandas().iloc[1]['metadata'])

vectorstore = MultimodalLanceDB(
    uri=LANCEDB_HOST_FILE, 
    embedding=embedder, 
    table_name=TBL_NAME
)

retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": 5}
)

query1 = "cat"
results1 = retriever.invoke(query1)
print(results1)

query2 = "cats are listening to the dogs barking"
results2 = retriever.invoke(query2)
print(results2)

query3 = "picture of motorcycle with the dogs barking around"
results3 = retriever.invoke(query3)
print(results3)

pd.iloc[100]['text']
pd.iloc[100]['transcript_for_inference']

res = retriever.invoke("what dessert is included in the video?")
len(res)
res[1]
res[0]

import whisper

model = whisper.load_model("small")
options = dict(task="translate", best_of=1, language='en')
results = model.transcribe("/home/tiep/projects/demo-cvpr-tutorial/src/tmp_videos/tmp_vid.mp3", **options)

from mcp_servers.bridgetower_search.utils.utils import getSubs

a = getSubs(results['segments'], "vtt")

res[0].metadata




from mcp_servers.bridgetower_search.openvino_bridgetower.openvino_bridgetower import OpenVINOBridgeTower
import requests
from PIL import Image
import lancedb
from mcp_servers.bridgetower_search.embeddings.bridgetower_embeddings import BridgeTowerEmbeddings
from mcp_servers.bridgetower_search.vectorstores.multimodal_lancedb import MultimodalLanceDB
LANCEDB_HOST_FILE = "./lancedb_vectorstore/.lancedb"
TBL_NAME = "mmrag"
db = lancedb.connect(LANCEDB_HOST_FILE)

embedder = BridgeTowerEmbeddings(
    text_vision_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_large_itc.xml",
    text_model_path="/home/tiep/projects/cvpr-tutorial/videoSearch/openVino/bridgetower-videosearch/models/custombridgetower_text_large_itc.xml",
    vision_model_path="TBD",
)


from typing import List, Tuple
import base64
from mcp import types
def search_from_video(
    query: str,   
    top_k: int = 2
) -> List[Tuple[str, str]]:
    """
    Search for relevant frames and transcripts from the video vectorstore.

    Args:
        query (str): The search query.

    Returns:
        List[Tuple[str, str]]: List of top_k retrieved byte64-encoded image and associated transcript.
    """
    # await ctx.info(f"Searching from video with query: {query} and top_k: {top_k}")
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE, 
        embedding=embedder, 
        table_name=TBL_NAME
    )

    retriever = vectorstore.as_retriever(
        search_type='similarity', 
        search_kwargs={"k": top_k}
    )
    
    results = retriever.invoke(query)
    response = []
    if results:        
        # await ctx.info(f"Found {len(results)} results for the query.")
        for result in results:
            frame_path = result.metadata.get('extracted_frame_path', None)
            transcript = result.metadata.get('transcript_for_inference', None)
            if frame_path and transcript:
                with open(frame_path, "rb") as f:
                    image_bytes = base64.b64encode(f.read()).decode("utf-8")
                b64_image = types.ImageContent(
                    type="image",
                    mimeType=frame_path.split(".")[-1],
                    data=image_bytes
                )
                
                response.append((b64_image, transcript))
    if not response:
        # await ctx.info("No results found for the query.")
        return types.CallToolResult(
            isError=True,
            content=[
                types.TextContent(
                    type="text",
                    text=f"Error: No results found for the query."
                )
            ]
        )
    # await ctx.info(f"Returning {len(response)} results.")
    return response

res1 = search_from_video(
    query="what dessert is included in this video?")

from mcp_servers.vlm_inference.openvino_multimodal import vlm
from mcp_servers.vlm_inference.utils import retrival_responses_to_qa_tmpl_str
prompt_str, images = retrival_responses_to_qa_tmpl_str(res1, "what dessert is included in this video?")
res2 = vlm.complete(
    prompt=prompt_str,
    image_documents=images,
    max_new_tokens=100,
    skip_prompt=True,
    # skip_special_tokens=True,
)
res2

res3 = vlm.stream_complete(
    prompt=prompt_str,
    image_documents=images,
    skip_prompt=True,
)
for r in res3:
    print(r.delta, end='')

from mcp_servers.vlm_inference.openvino_multimodal import vlm1,  vlm_processor

qa_tmpl_str = (
    "{image_holder}\n"
    "Given the provided information, including relevant frames as images and their corresponding contexts (their transcripts) extracted from the video, \
 accurately and precisely answer the query without any additional prior knowledge.\n"
    "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
    "Your answer should be consice.\n"
    "Only answer the query. Do NOT add anything else.\n"
    "---------------------\n"
    "{retrieved_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

image_holder = ""
retrieved_str = ""
for i, (img_doc, transcript) in enumerate(res1, start=1):
    image_holder += f"<|image_{str(i)}|>\n"
    retrieved_str += f"Context for image {i}: {transcript}\n"

new_prompt_str = qa_tmpl_str.format(
    image_holder=image_holder,
    retrieved_str=retrieved_str,
    query_str="what dessert is included in this video?"
)

messages = [
        {"role": "user", "content": prompt_str}
]  # Wrap conversation in a user role
    # Apply a chat template to format the message with the processor
text_prompt = vlm_processor.tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

inputs = vlm_processor(text=text_prompt, images=images, return_tensors="pt")
from transformers import TextStreamer
streamer=TextStreamer(vlm_processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
generate_ids = vlm1.generate(**inputs, max_new_tokens=100, streamer=streamer)
# vlm_processor.tokenizer.batch_decode(generate_ids[0], skip_special_tokens=True, skip_prompt=True)
output = ""
for i in streamer:
    output += i

from transformers import TextIteratorStreamer
streamer = TextIteratorStreamer(
    vlm_processor.tokenizer, skip_prompt=True, skip_special_tokens=True
)
generate_ids = vlm1.generate(
    **inputs,
    max_new_tokens=100,
    streamer=streamer,
)
output = ""
for i in streamer:
    output += i
print(output)

from llama_index.core.agent.workflow import ReActAgent, FunctionAgent

from llama_index.core.agent.workflow import AgentWorkflow

from llama_index.core.agent.workflow import multi_agent_workflow