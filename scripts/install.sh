#!/bin/bash
# Install necesssary dependencies

pip install -r requirements.txt
sudo apt-get install ffmpeg mediainfo sox libsox-fmt-mp3
add-apt-repository -y ppa:savoury1/ffmpeg4
apt-get -qq install -y ffmpeg
export HYDRA_FULL_ERROR=1