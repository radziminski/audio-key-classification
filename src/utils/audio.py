import audiofile
import os


def split_to_intervals(iut_filename, output_filename_prefix, interval_length):
    signal, sample_rate = audiofile.read(iut_filename)
    is_stereo = len(signal.shape) > 1

    samples_num = signal.shape[0]
    if is_stereo:
        samples_num = signal.shape[1]

    interval_samples_num = interval_length * sample_rate
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

    os.remove(iut_filename)


def split_to_intervals_in_dirs(directory, interval_length):
    for sub_directory in os.listdir(directory):
        sub_directory_path = os.path.join(directory, sub_directory)
        if os.path.isdir(sub_directory_path):
            for index, filename in enumerate(os.listdir(sub_directory_path)):
                filename_path = os.path.join(sub_directory_path, filename)
                if os.path.isfile(filename_path):
                    output_path = os.path.join(sub_directory_path, str(index))
                    split_to_intervals(filename_path, output_path, interval_length)
