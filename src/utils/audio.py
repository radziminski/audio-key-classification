import audiofile
import os
import scipy.signal
import torch
import audiofile
import numpy as np


def split_to_intervals(filename, output_filename_prefix, interval_length):
    signal, sample_rate = audiofile.read(filename)
    is_stereo = len(signal.shape) > 1

    samples_num = signal.shape[0]
    if is_stereo:
        samples_num = signal.shape[1]

    interval_samples_num = interval_length * sample_rate

    if samples_num == interval_samples_num:
        return

    intervals_num = samples_num // interval_samples_num

    for interval_index in range(intervals_num):
        interval_start = interval_index * interval_samples_num
        interval_end = interval_start + interval_samples_num

        interval = None
        if is_stereo:
            interval = signal.mean(axis=0)[interval_start:interval_end].reshape(1, -1)
        else:
            interval = signal[interval_start:interval_end].reshape(1, -1)

        audiofile.write(
            f"{output_filename_prefix}-{interval_index}.wav",
            interval,
            sample_rate,
            normalize=True,
        )

    os.remove(filename)


def split_to_intervals_in_dirs(directory, interval_length, extensions):
    for sub_directory in os.listdir(directory):
        sub_directory_path = os.path.join(directory, sub_directory)
        if os.path.isdir(sub_directory_path):
            for index, filename in enumerate(os.listdir(sub_directory_path)):
                parts = filename.split(".")
                extension = "." + parts[-1]

                if not extension in extensions:
                    continue

                filename_path = os.path.join(sub_directory_path, filename)
                if os.path.isfile(filename_path):
                    output_path = os.path.join(sub_directory_path, str(index))
                    split_to_intervals(filename_path, output_path, interval_length)


# old resample function that was too slow
def resample_old(audio, sr, target_sr):
    number_of_samples = round(len(audio) * float(target_sr) / sr)
    resampled_audio = scipy.signal.resample(audio, number_of_samples)

    return resampled_audio


def resample(audio, sr, target_sr):
    print(audio.shape)
    audio_length = len(audio)
    new_audio_length = audio_length // sr * target_sr
    resampled_audio = np.interp(
        np.linspace(0, 1, new_audio_length), np.linspace(0, 1, audio_length), audio
    )

    return resampled_audio


def common_audio_transform(sample, transform, target_sr):
    audio, sr = sample

    # Resample audio to target_sr (44100) sample rate, so that all inputs have the same size
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)

    audio_tensor = torch.tensor(audio, dtype=torch.float)

    if transform is not None and callable(transform):
        spectrogram = transform(audio_tensor)
        return spectrogram

    return audio


def common_audio_loader(file):
    return audiofile.read(file)
