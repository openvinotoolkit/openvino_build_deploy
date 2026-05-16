# Doc2Prototype with OpenVINO

Doc2Prototype is a command-line MVP demo for turning technical documents into structured JSON and downstream prototype artifacts.

The current scope is intentionally narrow:

- API documentation image -> PaddleOCR-VL with OpenVINO -> structured endpoint JSON -> FastAPI skeleton
- Flowchart image -> PaddleOCR-VL with OpenVINO -> structured nodes/edges JSON -> Mermaid flowchart

The demo also writes a static visual report with timing charts, extracted structure diagrams, document layout overlays, and text-density heatmaps.

## Requirements

- Python 3.10-3.12
- OpenVINO-compatible CPU. GPU/NPU/AUTO can be selected with `--device` when available.
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

Run the flowchart scenario:

```bash
python main.py examples/flowchart_sample.png --task flowchart --device CPU --output-dir outputs/mvp_flow_image_smoke
```

Fast structure-only smoke tests can use the Markdown samples and do not require the OpenVINO model:

```bash
python main.py examples/api_doc_sample.md --task api_doc --output-dir outputs/mvp_api_text_smoke
python main.py examples/flowchart_sample.md --task flowchart --output-dir outputs/mvp_flow_text_smoke
```

## Outputs

Each run writes:

- `raw_parse.md`: PaddleOCR-VL parser output
- `structured.json`: structured schema output
- `generated_api.py` or `generated_flowchart.mmd`: downstream prototype artifact
- `metrics.svg`: pipeline timing chart
- `api_endpoints.svg` or `flowchart.svg`: extracted structure visualization
- `layout_overlay.png`: document layout overlay with OpenVINO watermark
- `text_heatmap.png`: text density heatmap with OpenVINO watermark
- `visual_report.html`: single-page report linking all artifacts
- `run.json`: machine-readable run metadata and timing

Example reports are included under:

- `outputs/mvp_api_image_smoke/visual_report.html`
- `outputs/mvp_flow_image_smoke/visual_report.html`

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
- The CLI exposes device selection with `--device CPU|GPU|NPU|AUTO`.
- Each run records model load time, OpenVINO inference time, structure extraction time, generation time, and total time.
- Visual outputs include OpenVINO watermarking.

## Notes

The downstream code generation path is deterministic by default so the MVP remains reproducible without downloading a second LLM. The `--code-model-path` option is reserved for a local code model path when a Coder model is prepared separately.
