# Document Q&A with PaddleOCR-VL and OpenVINO™

Intelligent document understanding and question answering system powered by PaddleOCR-VL and OpenVINO. Upload product manuals (PDF/scanned documents/spec sheets with tables), and the system automatically parses, indexes, and answers questions with source citations.

**Full pipeline**: PDF → PaddleOCR-VL OCR (OpenVINO) → Table-aware Chunking → Qwen3-Embedding (OpenVINO) → ChromaDB → Qwen3-1.7B LLM (OpenVINO GenAI) → Answers with `[doc_name p.page]` citations.

All inference runs on **OpenVINO** — no PyTorch or external API needed.

## Demo Output

```
Q1: A100 型号的工作温度范围是多少？
A:  A100 型号的工作温度范围是 -20~70℃ [spec_with_tables p.1]
⏱  embed=75ms  retrieve=1.4ms  llm=3397ms  total=3474ms  tps=10.3

Q2: A300 型号的额定功率是多少瓦？
A:  A300 型号的额定功率是 500W [spec_with_tables p.1]

Q4: GB/T 2423.1—2008 对应的国际标准编号是多少？
A:  IEC 60068-2-1:2007 [gb_t_2423_1 p.3]
```

| Metric | Value (CPU, Intel i5) |
|--------|----------------------|
| Embed query | 75 ms |
| Retrieve | 1.4 ms |
| LLM generate | 3,397 ms |
| **End-to-end per question** | **3,474 ms** |
| LLM throughput | 10.3 tok/s |

## Quick Launch

### Windows

[Download install.bat](setup/install.bat), then double-click it. The script will clone the repo, create a virtual environment, install dependencies, and run the demo automatically.

### Linux / macOS

```bash
wget https://raw.githubusercontent.com/openvinotoolkit/openvino_build_deploy/master/demos/doc_qna_demo/setup/install.sh
chmod +x install.sh && ./install.sh
```

## Manual Setup

Supported Python versions: 3.10, 3.11, 3.12

### Step 1. Install dependencies

```bash
cd demos/doc_qna_demo
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### Step 2. Run the demo

```bash
# Windows: set these environment variables first
# set PYTHONIOENCODING=utf-8
# set HF_HUB_DISABLE_SYMLINKS=1

python main.py
```

On first run, the Embedding model (~600 MB) and LLM model (~1 GB) are automatically downloaded from HuggingFace.

### Step 3. Customize

```bash
# Single question
python main.py --question "A100 工作温度是多少？"

# Switch device (CPU / GPU / AUTO)
python main.py --device GPU

# Use a different LLM
python main.py --llm_model_id OpenVINO/Qwen3-8B-int4-ov
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PDF / Scanned Documents / Spec Sheets                      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  PaddleOCR-VL (OpenVINO)                                     │
│  Vision-language OCR → Structured Markdown                   │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Table-aware Chunking                                        │
│  Table rows with headers + semantic paragraph splitting      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Qwen3-Embedding-0.6B-int8 (OpenVINO)                       │
│  1024-dim multilingual embeddings → ChromaDB                 │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Qwen3-1.7B-int4 (OpenVINO GenAI)                           │
│  RAG generation with source citations [doc p.page]           │
└──────────────────────────────────────────────────────────────┘
```

## Models

All models use pre-converted OpenVINO IR from HuggingFace (auto-downloaded on first run):

| Role | Model | Size | Device |
|------|-------|------|--------|
| OCR | [PaddleOCR-VL-1.5-ov](https://huggingface.co/zhaohb/PaddleOCR-VL-1.5-ov) | ~2.7 GB | CPU/GPU |
| Embedding | [Qwen3-Embedding-0.6B-int8-ov](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) | ~600 MB | CPU/GPU |
| LLM | [Qwen3-1.7B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov) | ~1 GB | CPU/GPU |

**Disk requirement**: ~8 GB total (models + venv + ChromaDB).

## Key Features

- **Table-aware chunking**: Each table row carries its header as context, enabling precise cell-level lookup (e.g., "A300 rated power → 500W")
- **Anti-hallucination**: System prompt enforces strict grounding — answers only from retrieved context, refuses when evidence is absent
- **Source citations**: Every answer includes `[doc_name p.page]` traceable to the original document
- **All-OpenVINO inference**: OCR, Embedding, and LLM all run through OpenVINO — no PyTorch dependency at inference time
- **CPU-friendly**: Full pipeline runs at ~3.5s/question on Intel i5 (CPU-only)

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--question` | — | Single question (overrides `--questions_file`) |
| `--questions_file` | `data/demo_questions.txt` | One question per line |
| `--device` | `CPU` | OpenVINO device: CPU / GPU / AUTO |
| `--embed_model_id` | `OpenVINO/Qwen3-Embedding-0.6B-int8-ov` | Embedding model |
| `--llm_model_id` | `OpenVINO/Qwen3-1.7B-int4-ov` | LLM model |
| `--top_k` | `5` | Number of retrieved chunks |
| `--max_new_tokens` | `384` | Max generation length |
| `--out` | `results/demo_run.json` | JSON output path |
| `--out_md` | `results/demo_run.md` | Markdown report path |

## Known Limitations

1. **Conservative refusal on short answers**: The 1.7B LLM may refuse to answer when the retrieved evidence is a brief referral (e.g., "contact after-sales service") rather than a detailed procedure. Larger models (7B+) handle this better.
2. **Retrieval miss on semantically diluted chunks**: When the target fact is buried in a chunk dominated by other content (e.g., English headers), the small embedding model may miss it. Mitigation: increase `--top_k` or add a reranker.

## License

This demo is provided "as is" for demonstration and educational purposes.
