import argparse
import logging as log
import os

from typing import List, Set
from pathlib import Path
from tqdm import tqdm

import numpy as np
import openvino as ov
import yaml

from datasets import load_dataset
from urllib.request import getproxies
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openvino import OpenVINOLLM
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from transformers import AutoTokenizer

proxies = getproxies()
os.environ["http_proxy"]  = proxies["http"]
os.environ["https_proxy"] = proxies["https"]
os.environ["no_proxy"]    = "localhost, 127.0.0.1/8, ::1"
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig


DATASET_MAPPING = {
    "agribot_personality.yaml": {"name": "KisanVaani/agriculture-qa-english-only", "split": "train", "col": "question"},
    "healthcare_personality.yaml": {"name": "medalpaca/medical_meadow_medical_flashcards", "split": "train", "col": "input"},
    "bartender_personality.yaml": {"name": str(Path(__file__).parent / "bartender_personality.txt"), "col": "text"},
    "culinara_personality.yaml": {"name": str(Path(__file__).parent / "culinara_personality.txt"), "col": "text"},
    "tutor_personality.yaml": {"name": str(Path(__file__).parent / "tutor_personality.txt"), "col": "text"}
}
MODEL_DIR = Path("model")


def get_available_devices() -> Set[str]:
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}


def compute_deepeval_hallucination(inputs, outputs, contexts) -> float:
    avg_score = 0.
    for input, output, context in zip(inputs, outputs, contexts):
        test_case = LLMTestCase(
            input=input,
            actual_output=output,
            context=context
        )
        metric = HallucinationMetric(threshold=0.5)
        metric.measure(test_case)
        score = metric.score
        # reason = metric.reason
        avg_score += score / len(inputs)
    return avg_score


def prepare_dataset_and_model(chat_model_name: str, personality_file_path: Path, auth_token: str):
    dataset_info = DATASET_MAPPING.get(personality_file_path.name, "")
    assert dataset_info != ""
    log.info("Loading dataset")
    if dataset_info["name"].endswith(".txt"):
        dataset = load_dataset("text", data_files={"data": dataset_info["name"]})["data"]
    else:
        dataset = load_dataset(dataset_info["name"])[dataset_info["split"]]
    log.info("Dataset loading is finished")

    with open(personality_file_path, "rb") as f:
        chatbot_config = yaml.safe_load(f)

    ov_llm = load_chat_model(chat_model_name, auth_token)
    ov_chat_engine = SimpleChatEngine.from_defaults(llm=ov_llm, system_prompt=chatbot_config["system_configuration"],
                                                memory=ChatMemoryBuffer.from_defaults())
    return dataset[dataset_info["col"]], ov_chat_engine


def load_chat_model(model_name: str, token: str = None) -> OpenVINOLLM:
    model_path = MODEL_DIR / model_name

    # tokenizers are disabled anyway, this allows to avoid warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if token is not None:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": ""}
    # load llama model and its tokenizer
    if not model_path.exists():
        log.info(f"Downloading {model_name}... It may take up to 1h depending on your Internet connection and model size.")

        chat_tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        chat_tokenizer.save_pretrained(model_path)

        # openvino models are used as is
        is_openvino_model = model_name.split("/")[0] == "OpenVINO"
        if is_openvino_model:
            chat_model = OVModelForCausalLM.from_pretrained(model_name, export=False, compile=False, token=token)
            chat_model.save_pretrained(model_path)
        else:
            log.info(f"Loading and quantizing {model_name} to INT4...")
            log.info(f"Quantizing {model_name} to INT4... It may take significant amount of time depending on your machine power.")
            quant_config = OVWeightQuantizationConfig(bits=4, sym=False, ratio=0.8, quant_method="awq", group_size=128, dataset="wikitext2")
            chat_model = OVModelForCausalLM.from_pretrained(model_name, export=True, compile=False, quantization_config=quant_config,
                                                            token=token, trust_remote_code=True, library_name="transformers")
            chat_model.save_pretrained(model_path)

    device = "GPU" if "GPU" in get_available_devices() else "CPU"
    return OpenVINOLLM(context_window=4096, model_id_or_path=str(model_path), max_new_tokens=1024, device_map=device,
                       model_kwargs={"ov_config": ov_config, "library_name": "transformers"}, generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95})


def run_test_deepeval(chat_model_name: str, personality_file_path: Path, auth_token: str, selection_num: int = 10) -> float:
    """
    Args:
        chat_model_name (str): large language model path.
        personality_file_path (Path): personality file path.
        auth_token (str): auth token used for huggingface.
        selection_num (int): maximum number of prompt are selected to compute hallucination score

    Returns:
        hallucination score: the higher the score, the higher possibility of having hallucination issue.
    """
    dataset_question, ov_chat_engine = prepare_dataset_and_model(chat_model_name, personality_file_path, auth_token)
    inputs = dataset_question
    # We use question as context because the dataset lacks context
    contexts = dataset_question
    contexts_res = [[context] for context in contexts]

    outputs = []
    for input in tqdm(inputs[:selection_num]):
        output = ov_chat_engine.chat(input).response
        outputs.append(output)

    final_score = compute_deepeval_hallucination(inputs[:selection_num], outputs[:selection_num], contexts_res[:selection_num])
    return final_score


class OVSelfCheckLLMPrompt(SelfCheckLLMPrompt):
    def __init__(self, ov_chat_engine: SimpleChatEngine):
        self.ov_chat_engine = ov_chat_engine
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.not_defined_text = set()
        self.generate_num = 3

    def generate_outputs(self, prompt_list: List[str]) -> List[str]:
        response_list = []
        for prompt in tqdm(prompt_list, desc="generating responses"):
            tmp_list = []
            for _ in range(self.generate_num):
                response = self.ov_chat_engine.chat(prompt).response
                # remove </think> part
                response = response[response.rfind("</think>") + 8:].strip()
                tmp_list.append(response)
            response_list.append(tmp_list)
        return response_list

    def predict(
        self,
        sampled_passages: List[str],
    ) -> np.array:
        num_samples = len(sampled_passages)
        scores = np.zeros((num_samples, num_samples))

        for sent_i in range(num_samples):
            sentence = sampled_passages[sent_i].replace("\n", " ")
            for sample_i, sample in enumerate(sampled_passages):
                if sent_i == sample_i:
                    continue

                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ")
                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                generate_output = self.ov_chat_engine.chat(prompt).response

                # get text after </think>
                truncate_output = generate_output[generate_output.rfind("</think>") + 8:].strip()
                score_ = self.text_postprocessing(truncate_output)
                scores[sent_i, sample_i] = score_

        avg_score = np.sum(scores) / num_samples / (num_samples - 1)
        return avg_score


def run_test_selfcheckgpt(chat_model_name: str, personality_file_path: Path, auth_token: str, selection_num: int = 10) -> float:
    """
    Args:
        chat_model_name (str): large language model path.
        personality_file_path (Path): personality file path.
        auth_token (str): auth token used for huggingface.
        selection_num (int): maximum number of prompt are selected to compute hallucination score

    Returns:
        hallucination score: the higher the score, the higher possibility of having hallucination issue.
    """
    dataset_question, ov_chat_engine = prepare_dataset_and_model(chat_model_name, personality_file_path, auth_token)
    check_eng = OVSelfCheckLLMPrompt(ov_chat_engine)
    response_list = check_eng.generate_outputs(dataset_question[:selection_num])
    score_list = []
    for response_list_per_prompt in tqdm(response_list, desc="predict hallucination ratio"):
        score_list.append(check_eng.predict(response_list_per_prompt))
    final_score = float(np.mean(score_list))
    return final_score


if __name__ == "__main__":
    # set up logging
    log.getLogger().setLevel(log.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Path/name of the chat model")
    parser.add_argument("--personality", type=str, default="../healthcare_personality.yaml", help="Path to the YAML file with chatbot personality")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token to get Llama3")
    parser.add_argument("--check_type", type=str, choices=["deepeval", "selfcheckgpt"], default="deepeval", help="Hallucination check type")
    parser.add_argument("--selection_num", type=int, default=5, help="Maximum number of prompt are selected to compute hallucination score")

    args = parser.parse_args()
    if args.check_type == "deepeval":
        hallucination_score = run_test_deepeval(args.chat_model, Path(args.personality), args.hf_token, args.selection_num)
    else:
        hallucination_score = run_test_selfcheckgpt(args.chat_model, Path(args.personality), args.hf_token, args.selection_num)
    print(f"hallucination_score for personality {args.personality}: {hallucination_score}")
