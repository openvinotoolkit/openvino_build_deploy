#!/bin/bash
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python run_app.py
