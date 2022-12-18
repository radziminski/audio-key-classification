# Audio Key Classification

Project with full setup for training a deep-learning model for music key classification of a given audio file.

## Usage

TODO

## Available scripts

TODO...

## Data

Model is designed to classify spectrograms created from audio files.
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

## Testing

TODO...

## Training

TODO...

## Evaluation

TODO...
