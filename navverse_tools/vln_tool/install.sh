#!/bin/bash
conda install -y -c conda-forge python=3.10 "numpy<2.0.0" fastapi uvicorn python-multipart jinja2 pydantic
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip installlitellm scipy python-dotenv matplotlib
conda install -y openh264 x264 ffmpeg opencv-python -c conda-forge