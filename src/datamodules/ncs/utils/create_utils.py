from asyncore import write
import re
import urllib.request
import json
import os
from sklearn.model_selection import train_test_split
import zipfile


# NCS_PAGES_NUM = 60
NCS_PAGES_NUM = 3
BEATPORT_PAGES_NUM = 9

NCS_DATA_URL_REGEX = r"data-url=\".*?.mp3\""
NCS_DATA_ARTIST_REGEX = r"data-artist=\".*?\""
NCS_TRACK_REGEX = r"data-track=\".*?\""
REMOVE_TAGS_REGEX = r"<.*?>"

BEATPORT_TITLE_REGEX = r"class=\"buk-track-primary-title\" title=\".*?\""
BEATPORT_KEY_REGEX = r"<p class=\"buk-track-key\">.*?<\/p>"

BEATPORT_DATA_FILE = "beatport.json"

NOTES_LIST = ["C", "D", "E", "F", "G", "A", "B"]


def get_note_scales(note):
    scales = [f"${note} maj", f"${note} min"]

    if not note in ["C", "F"]:
        scales.append(f"${note}b maj", f"${note}b min")

    return scales


def map_scale_to_bemol(scale):
    if not "♯" in scale:
        return scale

    note = scale[0]
    scale_type = scale.split(" ")[1]
    note_index = NOTES_LIST.index(note)

    new_note = "C"
    if note_index < len(NOTES_LIST) - 1:
        new_note = NOTES_LIST[note_index + 1]

    new_scale = f"{new_note}b"
    if new_note == "C":
        new_scale = "C"
    elif new_note == "F":
        new_scale = "E"

    return f"{new_scale} {scale_type}"


def download_file(url, destination):
    try:
        print(f"Downloading: {url}...")
        urllib.request.urlretrieve(url, destination)
    except:
        print(f"Error downloading a file: {url}")
        return -1


def get_html(url):
    html = urllib.request.urlopen(url)
    html_bytes = html.read()

    html_string = html_bytes.decode("utf8")
    html.close()

    return html_string


def parse_beatport_page_for_songs(html_string):
    def parse_name(name_str):
        return name_str.split('"')[3].replace("&#39;", "'")

    def parse_key(key_str):
        without_tags = re.sub(REMOVE_TAGS_REGEX, "", key_str)
        return without_tags.replace("♭", "b")

    names_matches = re.findall(BEATPORT_TITLE_REGEX, html_string)
    names = list(map(parse_name, names_matches))

    keys_matches = re.findall(BEATPORT_KEY_REGEX, html_string)
    keys = list(map(parse_key, keys_matches))

    output_dicts = []
    for index in range(len(names)):
        output_dicts.append({"name": names[index], "key": keys[index]})

    return output_dicts


def get_beatport_data(destination_dir):
    songs = []
    beatport_data_filename = f"{destination_dir}/{BEATPORT_DATA_FILE}"
    try:
        beatport_data_file = open(beatport_data_filename, "r")
        beatport_data = json.load(beatport_data_file)
        beatport_data_file.close()

        return beatport_data
    except:
        print("Downloading beatport data...")
        for page in range(BEATPORT_PAGES_NUM):
            html = get_html(
                f"https://www.beatport.com/label/ncs/48816/tracks?page={page + 1}&per-page=150"
            )
            new_songs = parse_beatport_page_for_songs(html)
            for new_song in new_songs:
                songs.append(new_song)

        beatport_data_file = open(beatport_data_filename, "w")
        json.dump(songs, beatport_data_file)

        return new_songs


def get_song_name(artist, name):
    return f"{artist} - {name}.mp3"


def parse_ncs_page_for_songs(html):
    def parse_url(url):
        return url.split('"')[1]

    def parse_artist(artist):
        part = artist.split('"')[1]
        return re.sub(REMOVE_TAGS_REGEX, "", part)

    def parse_track(track):
        return track.split('"')[1]

    urls_matches = re.findall(NCS_DATA_URL_REGEX, html)
    urls = list(map(parse_url, urls_matches))

    artists_matches = re.findall(NCS_DATA_ARTIST_REGEX, html)
    artists = list(map(parse_artist, artists_matches))

    tracks_matches = re.findall(NCS_TRACK_REGEX, html)
    tracks = list(map(parse_track, tracks_matches))

    songs = []
    for index in range(len(urls)):
        songs.append(
            {
                "url": urls[index],
                "artist": artists[index],
                "name": tracks[index],
            }
        )

    return songs


def download_ncs_songs(destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    beatport_data = get_beatport_data(destination_dir)

    songs = []

    for page in range(NCS_PAGES_NUM):
        body = get_html(f"https://ncs.io/music?page={page + 1}")
        parsed = parse_ncs_page_for_songs(body)
        for song in parsed:
            songs.append(song)

    saved_count = 0
    all_count = 0

    for song in songs:
        name = song["name"]
        url = song["url"]
        artist = song["artist"]
        beatport_song = next(
            (
                beatport_song
                for beatport_song in beatport_data
                if song["name"].startswith(beatport_song["name"])
            ),
            None,
        )

        if beatport_song == None:
            continue

        key = beatport_song["key"]

        if key == "None":
            continue

        song_key = map_scale_to_bemol(key)
        all_count += 1

        song_key_parsed = song_key.replace(" ", "-")
        song_folder = os.path.join(destination_dir, song_key_parsed)
        song_name = get_song_name(artist, name)
        song_path = os.path.join(song_folder, song_name)
        song_train_path = os.path.join(
            destination_dir, "train", song_key_parsed, song_name
        )
        song_test_path = os.path.join(
            destination_dir, "validation", song_key_parsed, song_name
        )

        # Checking if song was already downloaded or splitted to train/test dirs
        if (
            os.path.exists(song_path)
            or os.path.exists(song_train_path)
            or os.path.exists(song_test_path)
        ):
            continue

        if not os.path.exists(song_folder):
            os.makedirs(song_folder)

        saved_count += 1
        download_file(url, song_path)

    print(f"Downloaded {saved_count} new songs.")
    print(f"In total {all_count} songs.")


def split_files_to_train_test(root_dir, train_path, test_path, train_ratio=0.85):
    train_dir = os.path.join(root_dir, train_path)
    test_dir = os.path.join(root_dir, test_path)
    files = []

    for sub_directory in os.listdir(root_dir):
        sub_directory_path = os.path.join(root_dir, sub_directory)
        if os.path.isdir(sub_directory_path):
            for filename in os.listdir(sub_directory_path):
                filename_path = os.path.join(sub_directory_path, filename)
                if os.path.isfile(filename_path):
                    files.append((filename, sub_directory))

    all_num = len(files)

    if all_num < 2:
        return

    train_num = int(train_ratio * all_num)
    test_num = all_num - train_num

    train_files, test_files = train_test_split(files, test_size=test_num)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Move some files to train directory
    for filename, file_sub_directory in train_files:
        old_file_path = os.path.join(root_dir, file_sub_directory, filename)
        new_file_dir = os.path.join(train_dir, file_sub_directory)
        new_file_path = os.path.join(new_file_dir, filename)

        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)

        os.replace(old_file_path, new_file_path)

    # Move some files to test directory
    for filename, file_sub_directory in test_files:
        old_file_path = os.path.join(root_dir, file_sub_directory, filename)
        new_file_dir = os.path.join(test_dir, file_sub_directory)
        new_file_path = os.path.join(new_file_dir, filename)

        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)

        os.replace(old_file_path, new_file_path)

    # Delete empty directories

    for sub_directory in os.listdir(root_dir):
        sub_directory_path = os.path.join(root_dir, sub_directory)
        if os.path.isdir(sub_directory_path):
            if sub_directory == "train" or sub_directory == "validation":
                continue

            os.rmdir(sub_directory_path)


def zipdir(path, zip_file):
    for root, dirs, files in os.walk(path):
        for file in files:
            zip_file.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


def zip_dataset(dataset_path, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zipdir(dataset_path, zip_file)
