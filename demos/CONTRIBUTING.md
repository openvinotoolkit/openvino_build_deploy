# How to add or update a demo

The goal of any demo in this directory is to present OpenVINO as an optimization and inference engine for AI applications. These demos are often used at events to bring attention to Intel's booth, so each demo should be interactive, engaging, and good-looking. And, of course, it must be fast enough so people can see the results!

## Implementing

Rules:
- The demo must be standalone - no dependencies to other demos (dependency to utils is ok)
- The demo must be a Python script called `main.py`
- All dependencies must be pinned to specific, stable, and tested versions and provided in the corresponding `requirements.txt`
- If the demo is visual (produces any video/image output) it must add an OpenVINO watermark to the output video/image (see utils)
- The demo must provide a README file with the instructions on supporting python versions (recommended 3.10-3.12), installing the environment, setting up and running (+ changing the behavior if applicable)
- The demo should provide a nice UI (Gradio is preferred)
- Gradio demos must provide both `--local_network` and `--public` parameters
- Webcam demos must provide a `--stream` parameter
- The demo should use utils for playing video streams, downloading files, and adding the watermark
- The demo should work and be tested for Windows, Linux and macOS (it may be verified through Github Actions)

## Merging

All updates are to be provided as a PR, which then should be reviewed by original authors (update of existing demo) or demos owners ([@adrianboguszewski](https://github.com/adrianboguszewski), [@zhuo-yoyowz](https://github.com/zhuo-yoyowz)) in case of a new contribution.
