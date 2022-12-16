#!/bin/bash
# Run experiments with different feature maps

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    tags="[gs_mtg_only, nf_16, test_full]" \
    "datamodule/image/train_datasets=[gs_mtg_dataset]"

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=20 \
    tags="[gs_mtg_only, nf_24, test_full]" \
    "datamodule/image/train_datasets=[gs_mtg_dataset]"

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    tags="[ncs_only, nf_16, test_full]" \
    "datamodule/image/train_datasets=[ncs_train_dataset]"

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=20 \
    tags="[ncs_only, nf_24, test_full]" \
    "datamodule/image/train_datasets=[ncs_train_dataset]"