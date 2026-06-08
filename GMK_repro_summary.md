# GMK Intel Core Ultra Mini PC Reproduction Summary

## Machine And Environment

- Machine: `NUCBOX_EVO-T1`
- Windows user: `intel`
- Workspace: `C:\Users\intel\Documents\Doc2Prototype`
- Repository: `C:\Users\intel\Documents\Doc2Prototype\openvino_build_deploy`
- Demo: `C:\Users\intel\Documents\Doc2Prototype\openvino_build_deploy\demos\doc2prototype_demo`
- CPU: `Intel(R) Core(TM) Ultra 9 285H`
- Windows GPU display name: `Intel(R) Arc(TM) 140T GPU (32GB)`

## Git And Runtime

- Fork: `https://github.com/Dryoung95/openvino_build_deploy`
- PR: `https://github.com/openvinotoolkit/openvino_build_deploy/pull/548`
- Branch: `doc2prototype-mvp`
- Reproduction branch status: synced with `origin/doc2prototype-mvp` before this report was added
- Python: `Python 3.12.4`
- Git: `git version 2.54.0.windows.1`
- OpenVINO: `2026.1.0-21367-63e31528c62-releases/2026/1`
- `available_devices`: `['CPU', 'GPU', 'NPU']`
- OpenVINO device names:
  - `CPU`: `Intel(R) Core(TM) Ultra 9 285H`
  - `GPU`: `Intel(R) Arc(TM) Graphics (iGPU)`
  - `NPU`: `Intel(R) AI Boost`

## Setup Notes

The virtual environment and dependencies were installed successfully. `prepare_model.py --device CPU` completed after setting `PYTHONIOENCODING=utf-8` to avoid a Windows console encoding issue when printing Unicode status characters.

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe prepare_model.py --device CPU
```

Model files, virtual environments, caches, and large reproduction output directories are intentionally not committed.

## Baseline Reproduction Results

| Scenario | Command summary | Device | Status | total_time | OpenVINO inference | Structured output | Agent review |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| Text smoke | `main.py examples/api_doc_sample.md --task api_doc` | CPU | success | 0.015s | 0.000s | 5 endpoints | `pass` |
| API document image | `main.py examples/api_doc_sample.png --task api_doc --device CPU` | CPU | success | 22.800s | 16.254s | 5 endpoints | `pass` |
| Flowchart image | `main.py examples/flowchart_sample.png --task flowchart --device CPU` | CPU | success | 20.495s | 14.289s | 6 nodes / 5 edges | `pass` |
| API image probe | `main.py examples/api_doc_sample.png --task api_doc --device GPU` | GPU | ran, weak extraction | 25.871s | 17.615s | 0 endpoints | `needs_attention` |
| API image probe | `main.py examples/api_doc_sample.png --task api_doc --device GPU.0` | GPU.0 | ran, weak extraction | 24.265s | 15.817s | 0 endpoints | `needs_attention` |
| API image probe | `main.py examples/api_doc_sample.png --task api_doc --device AUTO` | AUTO | failed before fallback optimization | n/a | n/a | n/a | n/a |
| API image probe | `main.py examples/api_doc_sample.png --task api_doc --device GPU.1` | GPU.1 | failed, device not available | n/a | n/a | n/a | n/a |
| API image probe | `main.py examples/api_doc_sample.png --task api_doc --device NPU` | NPU | failed, model graph/plugin limitation | n/a | n/a | n/a | n/a |

## Hardware-Aware Fallback Validation

The CLI now keeps the requested device as the first attempt, then falls back to `--fallback-device CPU` when the requested device fails or produces no task-specific structure such as API endpoints or flowchart nodes. `--no-device-fallback` preserves strict benchmark behavior.

| Scenario | Requested device | Effective device | Fallback | total_time | OpenVINO inference | Structured output | Agent review |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| API text smoke | CPU | CPU | no | 0.001s | 0.000s | 5 endpoints | `pass` |
| API image CPU | CPU | CPU | no | 22.521s | 16.311s | 5 endpoints | `pass` |
| API image Intel iGPU probe | GPU.0 | CPU | yes | 42.134s | 16.295s | 5 endpoints | `pass` |
| API image AUTO probe | AUTO | CPU | yes | 39.405s | 16.580s | 5 endpoints | `pass` |
| API image NPU probe | NPU | CPU | yes | 24.435s | 16.631s | 5 endpoints | `pass` |
| API image exact GPU probe | GPU.1 | CPU | yes | 23.213s | 16.688s | 5 endpoints | `pass` |

Negative check:

```powershell
.\.venv\Scripts\python.exe main.py examples/api_doc_sample.png --task api_doc --device GPU.1 --no-device-fallback --output-dir outputs\gmk_opt_api_gpu1_no_fallback
```

Result: failed as expected and preserved the original `GPU.1` device error instead of falling back to CPU.

## Device Interpretation

- `CPU` is the primary reproducible delivery path for this PR.
- `GPU` / `GPU.0` are available as Intel iGPU probe paths on this GMK machine. They can run the model, but this sample produced weak OCR structure, so the review correctly reports `needs_attention` without fallback.
- `AUTO` currently reaches an OpenVINO AUTO plugin limitation in the stateful PaddleOCR-VL LLM path.
- `NPU` is visible through OpenVINO, but the current PaddleOCR-VL stateful/dynamic-shape graph is not compatible with this NPU path.
- `GPU.1` is not exposed on this GMK machine. The fallback path records the device error and completes the user-facing report through CPU.

## Screenshot Evidence

The GMK CPU reproduction report screenshot is committed at:

```text
demos/doc2prototype_demo/assets/gmk_opt_api_cpu_visual_report.png
```

It corresponds to the optimized API image CPU run and shows `pass` review status with five extracted endpoints.

## Conclusion

The GMK Intel Core Ultra mini PC reproduction confirms the minimum Doc2Prototype delivery loop:

```text
image input -> PaddleOCR-VL OpenVINO inference -> structured JSON -> downstream generation -> agent review -> visual_report.html
```

The API document image extracts 5 endpoints and passes downstream review on CPU. The flowchart image extracts 6 nodes and 5 edges and also passes downstream review on CPU. Optional Intel iGPU / AUTO / NPU probes are documented with hardware-aware CPU fallback so reviewers can reproduce the demo without the run stopping on unsupported or weak device paths.
