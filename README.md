# Audio Key Classification

Project with full setup for training a deep-learning model for music key classification of a given audio file.
The results of the training can be found on Weights & Biases platform: https://wandb.ai/radziminski/audio-key-classification

## Usage

Run `make install` to install all system and python requirements for this project.

## Available scripts

- `make help` - Show help
- `make clean` - Clean autogenerated files
- `make clean-logs` - Clean logs
- `make format` - Run pre-commit hooks
- `make install` - Install all dependencies
- `make create-ncs-dataset` - Runs script that scraps the ncs.io site and creates NCS dataset
- `make prepare-audio` - Download and prepare datasets with mp3 audio files
- `make prepare-ncs-audio` - Download and prepare NCS dataset
- `make prepare-gs-mtg-audio` - Download and prepare GS MTG dataset
- `make prepare-gs-key-audio` - Download and prepare GS KEY dataset
- `make prepare-images` - Download images with precomputed spectrograms for each dataset
- `make create-spectrograms` - Create and save spectrogram images from existing audio files in all datasets
- `make prepare-spectrograms` - Downloads audio files, creates and saves spectrograms images for all datasets
- `make eval` - Default eval (on song fragments)
- `make eval-full` - Evaluation the model on full songs
- `make eval-audio` - Eval the model on audio files
- `make eval-audio-ncs-only` - Eval the model on audio files and only NCS dataset
- `make eval-audio-gs_key-only` - Eval the model on audio filesand only GS Key dataset
- `make eval-images` - Eval the model on spectrograms images files
- `make eval-images-ncs-only` - Eval the model with images and only NCS dataset
- `make eval-images-gs_mtg-only` - Eval the model with images and only GS Key dataset
- `make train` - Train the model
- `make train-from-checkpoint` - Train the model from checkpoint
- `make train-audio` - Train the model with AudioDataModule
- `make train-audio-ncs-only` - Train the model with audio and only NCS dataset
- `make train-audio-gs_mtg-only` - Train the model with audio and only GS MTG dataset
- `make train-images` - Train the model with ImageDataModule
- `make train-images-ncs-only` - Train the model with images and only NCS dataset
- `make train-images-gs_mtg-only` - Train the model with images and only GS MTG dataset
- `make test-default` - Test model by running 1 full epoch
- `make test-overfit` - Test model by running 1 train, val and test loop, using only 1 batch
- `make test-fdr` - Test model with overfit on 1 batch
- `make test-limit` - Test model by running train on 1% of data
- `make experiments-nf` - Run experiments with different number of feature maps
- `make experiments-hparams` - Run experiments for hyperparams tuning
- `make experiments-dataset` - Run experiments with different combinations of datasets
- `make debug` - Enter debugging mode with pdb

## Data

Model is designed to classify spectrograms created from audio files. It uses three datasets:

- NoCopyrightSounds - original (created by us) dataset based on freely available songs from [ncs.io](https://ncs.io).
- GiantSteps MTG - dataset with over 1000 2-minute parts of song from Beatport service with key corrections from service users
- GiantSteps Key - dataset with around 600 2-minute parts of song from Beatport service with key corrections from service users

There three main "modes" that model can be trained in:

### Audio

In this mode, model uses raw audio files for training/classification. Audio files are divided into intervals during data preparation. In data-loaders they are loaded into tensors and resampled into 44100 Hz sampling frequency. Then the Constant-Q transform (CQT) is applied to them, creating spectrograms. These spectrograms are the used for model training. This mode gives the biggest flexibility, since the Constant-Q parameters can be adjusted between each training. However, audio load and spectrogram creation takes significant time, resulting in very slow model training.

To use this mode you can run:

- `make prepare-audio` script for data preparation. It downloads three datasets, converts them to correct folder structure and splits each audio file into intervals.
- `make train-audio` script starts model training
- `make eval-audio` script starts model evaluation

### Audio/Image

In this mode, datasets with raw audio files are downloaded and then the spectrogram images are created from them in data preparation step (before the actual training). Then the model is trained directly on spectrogram images file. This mode gives little less flexibility, but it significantly speeds up training.

To use this mode you can run:

- `make prepare-spectrograms` script for data preparation. It downloads three audio datasets and saves spectrogram files from from every song intervals.
- `make train-image` script starts model training
- `make eval-image` script starts model evaluation

### Image

This mode is th same as the previous one, except that the pre-generated spectrogram images are downloaded and used for training. In this mode none of the spectrogram parameters can be adjusted, but its execution is fastest from all three modes.

To use this mode you can run:

- `make prepare-images` script for data preparation. It downloads three datasets with image spectrogram files inside.
- `make train-image` script starts model training
- `make eval-image` script starts model evaluation

## Training Testing

To test whether the model trains properly you can use one of tests scripts:

- `make test-default` - Test model by running 1 full epoch
- `make test-overfit` - Test model by running 1 train, val and test loop, using only 1 batch
- `make test-fdr` - Test model with overfit on 1 batch
- `make test-limit` - Test model by running train on 1% of data

## Training

You can train the model by running `src/train.py` file.

## Evaluation

You can train the model by running `src/eval.py` file.
