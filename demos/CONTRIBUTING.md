# How to add or update a demo

The goal of any demo in this directory is to present OpenVINO as an optimization and inference engine for AI applications. These demos are often used at events to bring attention to Intel's booth, so each demo should be interactive, engaging, and good-looking. 

## Implementing

Rules:
- The demo must be standalone - no dependencies to other demos (dependency to utils is ok)
- The demo must be a Python script (preferable one) called `main.py` or a Jupyter notebook
- All dependencies must be pinned to specific, stable, and tested versions and provided in the corresponding `requirements.txt` file (script) or the first code cell (notebook) 
- If the demo is visual (produces any video/image output) it must add an OpenVINO watermark to the output video/image (see utils)
- The demo must provide a README file with the instructions on installing the environment, setting up and running (+ changing the behavior if applicable)
- The demo should provide a nice UI (Gradio is preferred)
- The demo should use utils for playing video streams, downloading files, and adding the watermark
- The demo should work and be tested for Windows and Linux
- 
## Merging

All updates are to be provided as a PR, which then should be reviewed by original authors (update of existing demo) or demos owners ([@adrianboguszewski](https://github.com/adrianboguszewski), [@zhuo-yoyowz](https://github.com/zhuo-yoyowz)) in case of a new contribution.
