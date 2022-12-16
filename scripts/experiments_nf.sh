#!/bin/bash
# Run experiments with different feature maps

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=2 \
    tags="[full_dataset, nf_2, test_full]" \
    experiment_name='full-dataset-nf_2'

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=4 \
    tags="[full_dataset, nf_4, test_full]" \
    experiment_name='full-dataset-nf_4'

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=8 \
    tags="[full_dataset, nf_8, test_full]" \
    experiment_name='full-dataset-nf_8'

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=12 \
    tags="[full_dataset, nf_12, test_full]" \
    experiment_name='full-dataset-nf_12'

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    tags="[full_dataset, nf_16, test_full]" \
    experiment_name='full-dataset-nf_16'

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=20 \
    tags="[full_dataset, nf_20, test_full]" \
    experiment_name='full-dataset-nf_20'

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=30 \
    tags="[full_dataset, nf_30, test_full]" \
    experiment_name='full-dataset-nf_30'
