#!/bin/bash
# Run experiments with different feature maps

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    model.learning_rate=0.01 \
    model.optimizer.lr=0.01 \
    tags="[full_dataset, nf_16, lr_test]" 

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    model.learning_rate=0.1 \
    model.optimizer.lr=0.1 \
    tags="[full_dataset, nf_16, lr_test]" 

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    model.learning_rate=0.001 \
    model.optimizer.lr=0.001 \
    tags="[full_dataset, nf_16, lr_test]" 

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    model.optimizer.weight_decay=0.0001 \
    tags="[full_dataset, nf_16, wd_test]" 

python src/train.py \
    datamodule.image.batch_size=256 \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.model.num_feature_maps=16 \
    model.scheduler.factor=0.1 \
    tags="[full_dataset, nf_16, sch_test]" 