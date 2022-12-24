import audiofile
import os
import torch
import torchaudio
import numpy as np
import subprocess
import re
import datetime
import torchvision.transforms as transforms


def get_file_duration(filename):
    output = subprocess.run(
        ["ffmpeg", "-i", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stderr.decode()

    # Extract the duration from the output
    duration = re.search(r"Duration: (\d{2}:\d{2}:\d{2}\.\d{2})", output)
    if duration is None:
        return None
    duration = duration.group(1)
    time = datetime.datetime.strptime(duration, "%H:%M:%S.%f").time()
    # Get the total number of seconds
    seconds = time.second + 60 * time.minute

    return seconds


# replaced with mpeg subscript
def split_to_intervals_old(filename, output_filename_prefix, interval_length):
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


def split_to_intervals(filename, output_filename_prefix, interval_length):
    total_seconds = get_file_duration(filename)

    if total_seconds is None:
        os.remove(filename)
        return

    # Skipping files with correct length
    if abs(int(total_seconds) - interval_length) < 2:
        return

    cmd = f'ffmpeg -i "{filename}" -f segment -segment_time {interval_length} -reset_timestamps 1 -avoid_negative_ts make_zero -ac 1 -ar 44100 -c copy {output_filename_prefix}-%03d.mp3'
    proc = subprocess.run(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    rm_count = 0

    # Go through created files and delete the ones that are shorter than interval_length
    if proc.returncode == 0:
        for file in os.listdir(os.path.dirname(filename)):
            basename = os.path.basename(output_filename_prefix)
            if file.startswith(basename) and (
                file.endswith("mp3") or file.endswith("wav")
            ):
                filepath = os.path.join(os.path.dirname(filename), file)

                # Get the total number of seconds
                seconds = get_file_duration(filepath)

                # Removing last part of song with wrong duration
                if seconds is None or abs(int(seconds) - interval_length) > 1:
                    os.remove(filepath)
                    rm_count += 1

        if os.path.exists(filename):
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
                    output_path = os.path.join(sub_directory_path, f"akc-split-{index}")
                    split_to_intervals(filename_path, output_path, interval_length)


def resample(audio, sr, target_sr):
    audio_length = len(audio)
    new_audio_length = audio_length // sr * target_sr
    resampled_audio = np.interp(
        np.linspace(0, 1, new_audio_length), np.linspace(0, 1, audio_length), audio
    )

    return resampled_audio


# Old version without torchaudio
def common_audio_transform_old(sample, transform, target_sr, target_length, device):
    audio, sr = sample

    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)

    # Resample audio to target_sr (44100) sample rate, so that all inputs have the same size
    if sr != target_sr:
        audio = resample(audio, sr, target_sr)

    # FFMPEG does not cut the files into intervals perfectly - there are some minor
    # correction needed:
    if len(audio) < target_length:
        audio = np.pad(
            audio, (0, target_length - len(audio)), "constant", constant_values=(0, 0)
        )

    if len(audio) > target_length:
        audio = audio[:target_length]

    # TODO: investigate why device is needed here
    audio_tensor = torch.tensor(audio, dtype=torch.float, device=device)

    if transform is not None and callable(transform):
        spectrogram = transform.to(device)(audio_tensor)
        return spectrogram

    return audio


# potentially TODO - extract to hydra config
NORMALIZE_TRANSFORM = transforms.Normalize((0.94025,), (2.21655,))


# New version with torchaudio
def common_audio_transform(sample, transform, target_sr, target_length, device):
    audio, sr = sample

    if len(audio.shape) > 1:
        audio = audio.mean(axis=0)

    # Resample audio to target_sr (44100) sample rate, so that all inputs have the same size
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
        audio = resampler(audio)

    # FFMPEG does not cut the files into intervals perfectly - there are some minor
    # correction needed:
    if len(audio) < target_length:
        audio = torch.nn.functional.pad(audio, (0, target_length - len(audio))).to(
            device
        )

    if len(audio) > target_length:
        audio = audio[:target_length]

    new_audio = audio.clone().to(device)

    if transform is not None and callable(transform):
        spectrogram = transform.to(device)(new_audio)
        # spectrogram = NORMALIZE_TRANSFORM.to(device)(spectrogram)
        return spectrogram

    return new_audio


def common_audio_loader(file, type="torch", device="cuda"):
    # DEPRECATED !!! - don't use torchaudio loader, it needs outdated dependencies
    if type == "torch":
        audio, sr = torchaudio.load(file, format="mp3")
        audio.to(device)

        return audio, sr

    audio, sr = audiofile.read(file)
    audio = torch.tensor(audio, dtype=torch.float, device=device)

    return audio, sr


def try_delete_dir(dir):
    try:
        cmd = f"rm -rf {dir}"
        subprocess.run(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except:
        return
