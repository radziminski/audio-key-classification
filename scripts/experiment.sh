#!/bin/bash
python3 src/train.py \
trainer.max_epochs=30 \
model.model.num_feature_maps=16 \
"model/criterion=cross_entropy" \
"model/model=allconv" \
tags="[full_dataset, nf_16, test_full, cross_entropy, allconv, hop_2048]"

python3 src/train.py trainer.max_epochs=30 model.model.num_feature_maps=2 "model/criterion=cross_entropy" "model/model=allconv" tags="[full_dataset, nf_2, test_full, cross_entropy, allconv, hop_2048]"
