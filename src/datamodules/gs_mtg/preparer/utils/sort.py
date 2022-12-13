import os
import csv

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#" "B"]
MAJOR_SCALES = [f"{note} major" for note in NOTES]
MINOR_SCALES = [f"{note} minor" for note in NOTES]


def get_filename(id):
    return f"{id}.LOFI.mp3"


def get_filepath(id, audio_dir):
    return os.path.join(audio_dir, get_filename(id))


def sort_files_to_dirs(annotations_file, audio_dir):
    if not os.path.isdir(audio_dir):
        os.mkdir(audio_dir)
    with open(annotations_file, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        for row in csv_reader:
            id = row[0]
            key = row[1]
