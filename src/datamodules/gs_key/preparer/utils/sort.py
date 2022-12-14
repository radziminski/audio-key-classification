import os

NOTES = [
    "Ab",
    "A",
    "Bb",
    "B",
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
]

MAJOR_SCALES = [f"{note} major" for note in NOTES]
MINOR_SCALES = [f"{note} minor" for note in NOTES]
SCALES = [*MAJOR_SCALES, *MINOR_SCALES]

MAJOR_SCALES_B_SHORT = [f"{note}-maj" for note in NOTES]
MINOR_SCALES_B_SHORT = [f"{note}-min" for note in NOTES]
SCALES_B_SHORT = [*MAJOR_SCALES_B_SHORT, *MINOR_SCALES_B_SHORT]


def sort_files_to_dirs(audio_dir, keys_dir, dataset_dir):
    for scale_dir in SCALES_B_SHORT:
        scale_dir_full = os.path.join(dataset_dir, scale_dir)
        if not os.path.isdir(scale_dir_full):
            os.mkdir(scale_dir_full)

    songs_with_no_class = 0

    for file_index, key_file in enumerate(os.listdir(keys_dir)):
        if not key_file.endswith("txt"):
            continue

        key_file_full = os.path.join(keys_dir, key_file)
        audio_filepath = os.path.join(audio_dir, key_file.replace("txt", "mp3"))
        with open(key_file_full, "r") as file:
            key = file.read().replace("\n", "")

            if not os.path.exists(audio_filepath):
                continue

            has_class = False

            for index, scale in enumerate(SCALES):
                if scale in key:
                    scale_dir = SCALES_B_SHORT[index]
                    new_filepath = os.path.join(
                        dataset_dir, scale_dir, f"{file_index}.mp3"
                    )
                    os.rename(audio_filepath, new_filepath)
                    has_class = True
                    break

            if not has_class:
                songs_with_no_class += 1
                os.remove(audio_filepath)

    print(f"Skipped {songs_with_no_class} files without classes")
