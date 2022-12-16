#!/bin/bash
# Install necesssary dependencies

pip install -r requirements.txt
sudo apt-get install ffmpeg mediainfo sox libsox-fmt-mp3
export HYDRA_FULL_ERROR=1