import os
import csv

NOTES = [
    "G#",
    "A",
    "A#",
    "B",
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
]
MAJOR_SCALES = [f"{note} major" for note in NOTES]
MINOR_SCALES = [f"{note} minor" for note in NOTES]
SCALES = [*MAJOR_SCALES, *MINOR_SCALES]

NOTE_B = [
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
MAJOR_SCALES_B_SHORT = [f"{note}-maj" for note in NOTE_B]
MINOR_SCALES_B_SHORT = [f"{note}-min" for note in NOTE_B]
SCALES_B_SHORT = [*MAJOR_SCALES_B_SHORT, *MINOR_SCALES_B_SHORT]


def get_filename(id):
    return f"{id}.LOFI.mp3"


def get_filepath(id, audio_dir):
    return os.path.join(audio_dir, get_filename(id))


def sort_files_to_dirs(annotations_file, audio_dir, dataset_dir):
    for scale_dir in SCALES_B_SHORT:
        scale_dir_full = os.path.join(dataset_dir, scale_dir)
        if not os.path.isdir(scale_dir_full):
            os.mkdir(scale_dir_full)

    songs_with_no_class = 0

    with open(annotations_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        for index, row in enumerate(csv_reader):
            if index == 0:
                continue

            id = row[0]
            key = row[1]

            filepath = get_filepath(id, audio_dir)

            if not os.path.exists(filepath):
                continue

            has_class = False

            for index, scale in enumerate(SCALES):
                if scale in key:
                    scale_dir = SCALES_B_SHORT[index]
                    new_filepath = os.path.join(dataset_dir, scale_dir, f"{id}.mp3")
                    os.rename(filepath, new_filepath)
                    has_class = True
                    break

            if not has_class:
                songs_with_no_class += 1
                os.remove(filepath)

    print(f"Skipped {songs_with_no_class} files without classes")
