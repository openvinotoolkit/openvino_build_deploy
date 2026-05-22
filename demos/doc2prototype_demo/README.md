# Doc2Prototype with OpenVINO

Doc2Prototype is a command-line MVP demo for turning technical documents into structured JSON and downstream prototype artifacts.

The current scope is intentionally narrow:

- API documentation image -> PaddleOCR-VL with OpenVINO -> structured endpoint JSON -> FastAPI skeleton
- Flowchart image -> PaddleOCR-VL with OpenVINO -> structured nodes/edges JSON -> Mermaid flowchart

An optional official/reference sample path is included for sanity-checking general PaddleOCR-VL document recognition with a Markdown technical summary.

The demo also writes a static visual report with timing charts, extracted structure diagrams, document layout overlays, and text-density heatmaps.

## Requirements

- Python 3.10-3.12
- OpenVINO-compatible CPU. GPU/NPU/AUTO can be selected with `--device` when available. Use exact device IDs such as `GPU.0` or `GPU.1` when multiple GPU plugins are visible.
- Enough disk space for PaddleOCR-VL and converted OpenVINO IR files. Model files are not committed to this repository.

## Setup

```bash
cd openvino_build_deploy/demos/doc2prototype_demo
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare the OpenVINO Model

Run this once to download PaddleOCR-VL and export it to OpenVINO IR:

```bash
python prepare_model.py --device CPU
```

The converted model is written to `ov_paddleocr_vl_model/`. This directory is intentionally ignored by git.

## Run the MVP Demo

Generate clean sample PNG inputs:

```bash
python scripts/make_sample_images.py
```

Run the API documentation scenario:

```bash
python main.py examples/api_doc_sample.png --task api_doc --device CPU --output-dir outputs/mvp_api_image_smoke
```

For API document images, the default parser prompt is `OCR:`. The endpoint schema is then extracted deterministically from the OCR text, which is more stable than asking the vision-language model to directly infer the API schema.

Run the flowchart scenario:

```bash
python main.py examples/flowchart_sample.png --task flowchart --device CPU --output-dir outputs/mvp_flow_image_smoke
```

Fast structure-only smoke tests can use the Markdown samples and do not require the OpenVINO model:

```bash
python main.py examples/api_doc_sample.md --task api_doc --output-dir outputs/mvp_api_text_smoke
python main.py examples/flowchart_sample.md --task flowchart --output-dir outputs/mvp_flow_text_smoke
```

## Run an Official Reference Sample

The MVP scenarios above use project-specific API and flowchart inputs. To also show that the OpenVINO PaddleOCR-VL path works on an official/reference document image, prepare the sample image:

```bash
python scripts/prepare_official_sample.py
```

The script first tries to copy the OpenVINO Notebooks PaddleOCR-VL sample from `openvino_notebooks/notebooks/paddleocr_vl/test.png`. If that checkout is not available, it downloads the public PaddleOCR-VL demo image.

Run the reference sample as a generic technical document:

```bash
python main.py examples/official_paddleocr_vl_sample.png --task technical_doc --prompt "OCR:" --device CPU --output-dir outputs/mvp_official_reference
```

This produces structured section JSON, a Markdown technical summary, OpenVINO timing data, and the same visual report format used by the MVP scenarios.

## Outputs

Each run writes:

- `raw_parse.md`: PaddleOCR-VL parser output
- `structured.json`: structured schema output
- `generated_api.py` or `generated_flowchart.mmd`: downstream prototype artifact
- `agent_trace.json`: structured downstream agent workflow trace
- `agent_review.md`: downstream coverage review result
- `metrics.svg`: pipeline timing chart
- `api_endpoints.svg` or `flowchart.svg`: extracted structure visualization
- `document_summary.svg`: extracted section visualization for the optional technical document reference sample
- `layout_overlay.png`: document layout overlay with OpenVINO watermark
- `text_heatmap.png`: text density heatmap with OpenVINO watermark
- `visual_report.html`: single-page report linking all artifacts
- `run.json`: machine-readable run metadata and timing

Example reports are included under:

- `outputs/mvp_api_image_smoke/visual_report.html`
- `outputs/mvp_flow_image_smoke/visual_report.html`

## Effect Display

The tracked smoke reports demonstrate the complete path from document/visual understanding to downstream agent processing.

API document image to FastAPI skeleton:

![Doc2Prototype API report](assets/doc2prototype_api_report.png)

Flowchart image to Mermaid diagram:

![Doc2Prototype flowchart report](assets/doc2prototype_flowchart_report.png)

## Reviewer Quick Reproduction

From this demo directory, the following commands reproduce the two primary smoke reports used for PR review:

```bash
python prepare_model.py --device CPU
python scripts/make_sample_images.py
python main.py examples/api_doc_sample.png --task api_doc --device CPU --output-dir outputs/mvp_api_image_smoke
python main.py examples/flowchart_sample.png --task flowchart --device CPU --output-dir outputs/mvp_flow_image_smoke
```

The expected high-level results are:

| Scenario | OpenVINO | Device | Structured output | Downstream artifact | Agent review |
| --- | --- | --- | --- | --- | --- |
| API document image | yes | CPU | 5 endpoints | `generated_api.py` | pass |
| Flowchart image | yes | CPU | 6 nodes / 5 edges | `generated_flowchart.mmd` | pass |

## View Reports and Capture Screenshots

After running the demo, open `visual_report.html` in a browser to inspect the full result.

If the project is running in WSL on Windows, the example reports can be opened from File Explorer or a browser with:

```text
\\wsl.localhost\Ubuntu\root\Doc2Prototype\openvino_build_deploy\demos\doc2prototype_demo\outputs\mvp_api_image_smoke\visual_report.html
```

```text
\\wsl.localhost\Ubuntu\root\Doc2Prototype\openvino_build_deploy\demos\doc2prototype_demo\outputs\mvp_flow_image_smoke\visual_report.html
```

Recommended screenshots for reporting:

- API document report: run summary, pipeline timing, extracted API surface, layout overlay, and text heatmap.
- Flowchart report: run summary, pipeline timing, extracted flowchart, layout overlay, and text heatmap.
- Pull request page: PR number, open status, and successful checks.

On Windows, press `Win + Shift + S`, select the browser region, and save the screenshots for the weekly report.

## OpenVINO Value Shown

- PaddleOCR-VL inference runs through an OpenVINO IR model.
- The CLI exposes device selection with `--device CPU|GPU|GPU.0|GPU.1|NPU|AUTO`.
- Each run records model load time, OpenVINO inference time, structure extraction time, generation time, and total time.
- Visual outputs include OpenVINO watermarking.

## Notes

The downstream code generation path is deterministic by default so the MVP remains reproducible without downloading a second LLM. When a local Coder model is prepared separately, pass `--code-model-path` and select `--code-model-backend hf|openvino|auto`.

## Downstream Agent Flow

After PaddleOCR-VL parsing and structured JSON extraction, the CLI routes the result through a downstream agent workflow:

1. `PlannerAgent` summarizes the structured input and selects the downstream artifact target.
2. `GeneratorAgent` invokes the code/summary generator. By default this uses deterministic templates for reproducibility; when `--code-model-path` is provided, the same workflow can call a local OpenVINO/HF Coder model. Select the backend with `--code-model-backend openvino|hf|auto|template`.
3. `ReviewAgent` checks whether the generated artifact covers the extracted endpoints, flowchart nodes, or document sections.

Each run writes:

- `agent_trace.json`: machine-readable agent plan, steps, backend, and review status.
- `agent_review.md`: human-readable review of the downstream generation result.

This makes the required handoff from document understanding to downstream intelligent processing explicit and reproducible without forcing users to download a second LLM for the default MVP path.

Example local HuggingFace Coder run:

```bash
python -c "from code_generator import download_code_model; print(download_code_model('Qwen/Qwen2.5-Coder-0.5B-Instruct'))"
python main.py examples/api_doc_sample.md --task api_doc --code-model-path <printed-model-path> --code-model-backend hf --code-max-new-tokens 768 --output-dir outputs/mvp_api_text_hf_coder
```

For example, ModelScope may print a local path similar to `_models\Qwen\Qwen2___5-Coder-0___5B-Instruct` on Windows.

To validate the full image-to-Coder path:

```bash
python main.py examples/api_doc_sample.png --task api_doc --device CPU --code-model-path <printed-model-path> --code-model-backend hf --code-max-new-tokens 768 --output-dir outputs/mvp_api_image_hf_coder
```
