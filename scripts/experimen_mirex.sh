#!/bin/bash
python3 src/train.py \
    trainer.max_epochs=30 \
    model.model.num_feature_maps=16 \
    "model/criterion=mirex_v0" \
    "model/model=allconv" \
    tags="[full_dataset, nf_16, test_full, mirex_v0, allconv, hop_2048]"
