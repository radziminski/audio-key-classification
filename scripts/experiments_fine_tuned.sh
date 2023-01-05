#!/bin/bash
# Run experiments with different feature maps

python src/train.py \
    datamodule.image.batch_size=1024 \
    trainer=gpu \
    trainer.max_epochs=70 \
    model.learning_rate=0.09120108393559097 \
    model.optimizer.lr=0.09120108393559097 \
    model.model.num_feature_maps=2 \
    tags="[full_dataset, nf_2, test_full, fine_tuned params]" \
    experiment_name='full-dataset-nf_2-fine-tuned'