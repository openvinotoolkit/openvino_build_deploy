from transformers import AutoProcessor, AutoTokenizer
import os
from utils.utils import load_env
from llama_index.multi_modal_llms.openvino import OpenVINOMultiModal

from optimum.intel.openvino import OVModelForVisualCausalLM
from .utils import retrival_responses_to_qa_tmpl_str
from transformers import TextIteratorStreamer 

load_env()
vlm_model_path =  os.getenv("VLM_MODEL_PATH", None)
vlm_model_device = os.getenv("VLM_MODEL_DEVICE", "CPU")
vlm_processor = AutoProcessor.from_pretrained(vlm_model_path, trust_remote_code=True)
# vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_model_path)

def messages_to_prompt(messages, image_documents):
    """
    Prepares the input messages and images.
    """
    images = []
      # Add user text message
    for img_doc in image_documents:
        images.append(img_doc)
    messages = [
        {"role": "user", "content": messages[0].content}
    ]  # Wrap conversation in a user role
    # Apply a chat template to format the message with the processor
    text_prompt = vlm_processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Prepare the model inputs (text + images) and convert to tensor
    return vlm_processor(text=text_prompt, images=images, return_tensors="pt")


# Load the OpenVINO model directly instead of using LlamaIndex wrapper
# to avoid re-export issues with already converted models
print(f"[VLM MODEL] Initializing VLM model...")
print(f"[VLM MODEL] Model path: {vlm_model_path}")
print(f"[VLM MODEL] Target device: {vlm_model_device}")
vlm1 = OVModelForVisualCausalLM.from_pretrained(
    vlm_model_path,
    device=vlm_model_device,
    trust_remote_code=True,
)
print(f"[VLM MODEL] VLM model loaded successfully on {vlm_model_device} device")

# Comment out the LlamaIndex wrapper since we're using the direct OpenVINO model
# vlm = OpenVINOMultiModal(
#     model_id_or_path=vlm_model_path,
#     device_map=vlm_model_device,
#     messages_to_prompt=messages_to_prompt,
#     model_kwargs={"trust_remote_code": True},
#     generate_kwargs={"do_sample": False, "eos_token_id": vlm_processor.tokenizer.eos_token_id},
# )

def vlm_inference(retrieval_messages, query):
    """
    Perform inference using the OpenVINO VLM model.
    Args:
        retrieval_messages: List of retrieval messages with images and text.
        query (str): The query string to be answered by the model.
    Returns:
        str: The generated response from the model.
    """
    try:
        prompt, images = retrival_responses_to_qa_tmpl_str(
            retrieval_messages, query
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt = vlm_processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = vlm_processor(text=prompt, images=images, return_tensors="pt")
        
        # Use direct generation instead of streaming to avoid memory issues
        generate_ids = vlm1.generate(
            **inputs,
            max_new_tokens=200,  # Limit tokens to prevent memory overflow
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=vlm_processor.tokenizer.eos_token_id,
        )
        
        # Extract only the new tokens (skip the input prompt)
        new_tokens = generate_ids[0][inputs['input_ids'].shape[1]:]
        output = vlm_processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return output.strip()
        
    except Exception as e:
        print(f"VLM Inference Error: {e}")
        return f"Error during VLM inference: {str(e)}"
