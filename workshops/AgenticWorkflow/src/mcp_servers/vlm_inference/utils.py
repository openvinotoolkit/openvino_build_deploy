from typing import List, Tuple
import PIL
from llama_index.core.schema import ImageDocument

def retrival_responses_to_qa_tmpl_str(messages: List[Tuple[str, str]], query_str: str) -> Tuple[str, List[PIL.Image.Image]]:
    """
    Convert retrieval responses to a question-answer template string.

    Args:
        messages (List[Tuple[str, str]]): List of tuples containing question and answer pairs.

    Returns:
        str: Formatted string with questions and answers.
    """
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
    images = []
    for i, (b64_img, transcript) in enumerate(messages, start=1):
        image_holder += f"<|image_{str(i)}|>\n"
        retrieved_str += f"Context for image {i}: {transcript}\n"
        if isinstance(b64_img, PIL.Image.Image):
            images.append(b64_img)
        else:
            # Assuming b64_img is types.ImageContent
            from io import BytesIO
            import base64
            b64_img = b64_img.data
            image_data = base64.b64decode(b64_img)
            image = PIL.Image.open(BytesIO(image_data))
            images.append(image)    
    return qa_tmpl_str.format(
        image_holder=image_holder,
        retrieved_str=retrieved_str,
        query_str=query_str
    ), images