#!/bin/bash

cd audio-key-classification && python3 src/train.py test=overfit logger=tensorboard "model/model=allconv"