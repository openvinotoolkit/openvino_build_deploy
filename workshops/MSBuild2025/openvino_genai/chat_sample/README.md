# OpenVINO Chat Sample

Follow instructions here to prepare the environment:
https://github.com/raymondlo84Fork/MSBuild2025/blob/main/openvino_genai/README.md

```
#Make sure you activate the environment after restarting the terminal
./openvino_venv/Script/bin
```

## How to use a LLM model from HuggingFace

To download a pre-compressed model (for CPU/GPU only) and experiment with the latest Phi-4-mini-instruct model:
```
huggingface-cli download OpenVINO/Phi-4-mini-instruct-int4-ov --local-dir Phi-4-mini-instruct-int4-ov
```

To download and compress a model (CPU/GPU/NPU):
```
 optimum-cli export openvino -m microsoft/Phi-3-mini-4k-instruct  --trust-remote-code --weight-format int4 --sym --ratio 1.0 --group-size 128 Phi-3-mini-4k-instruct-npu
```
For NPU usage, please make sure the flags `--weight-format int4`, `--sym` and `--group-size 128` are set.

To test run NPU without a huge download, you can try TinyLlama:
```
optimum-cli export openvino -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 --sym --ratio 1.0 --group-size 128 TinyLlama-1.1B-Chat-v1.0
```

To obtain a meta llama demo, please first get a access token from this link [Access Security Tokens](https://huggingface.co/docs/hub/en/security-tokens), then login with the command line. Additionally, you have to accept to the agreement and wait for the approval (https://huggingface.co/meta-llama). Often this only take a few minutes to an hour.

```
huggingface-cli login
```
Then, you can execute this command to convert the model to be compatible with the NPU.
```
optimum-cli export openvino --model meta-llama/Llama-3.2-3B-Instruct  --trust-remote-code --task text-generation-with-past --weight-format int4 --group-size -1 --sym --ratio 1.0 llama-3.2-3b-instruct-INT4-npu
```

## How to Run

```
python chat_sample.py Phi-4-mini-instruct-int4-ov
```
or replace the model with `Phi-3-mini-4k-instruct-int4-npu` or `Llama-3.2-3B-Instruct-npu`.

By default, we enabled CPU in `chat_sample.py`. You can deploy the LLMs on GPU or NPU by simply replacing the device name as `GPU` or `NPU` in the code.
```
    device = 'CPU'  # GPU or NPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)
```

Llama 3.2 3B example output:
![Screenshot 2025-04-28 133741](https://github.com/user-attachments/assets/532f6d66-2cc4-4a29-b71c-9c15f3716e7e)

## References:
NPU with OpenVINO GenAI: https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html

