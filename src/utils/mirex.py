import torch

keys_idx_map = {
    0: "11B",
    1: "8A",
    2: "4B",
    3: "1A",
    4: "1B",
    5: "10A",
    6: "6B",
    7: "3A",
    8: "8B",
    9: "5A",
    10: "10B",
    11: "7A",
    12: "3B",
    13: "12A",
    14: "12B",
    15: "9A",
    16: "5B",
    17: "2A",
    18: "7B",
    19: "4A",
    20: "9B",
    21: "6A",
    22: "2B",
    23: "11A",
}

keys_map = {
    "1A": "Ab minor",
    "2A": "Eb minor",
    "3A": "Bb minor",
    "4A": "F minor",
    "5A": "C minor",
    "6A": "G minor",
    "7A": "D minor",
    "8A": "A minor",
    "9A": "E minor",
    "10A": "B minor",
    "11A": "Gb minor",
    "12A": "Db minor",
    "1B": "B major",
    "2B": "Gb major",
    "3B": "Db major",
    "4B": "Ab major",
    "5B": "Eb major",
    "6B": "Bb major",
    "7B": "F major",
    "8B": "C major",
    "9B": "G major",
    "10B": "D major",
    "11B": "A major",
    "12B": "E major",
}
keys_map_inv = {v: k for k, v in keys_map.items()}
keys_idx_map_inv = {v: k for k, v in keys_idx_map.items()}


def get_maps():
    return keys_idx_map, keys_idx_map_inv


def mirex_score(true_key_indices, predicted_key_indices):
    score = torch.empty(size=true_key_indices.shape, device="cuda")
    for i, true in enumerate(true_key_indices):
        single_score = mirex_score_single(true.item(), predicted_key_indices[i].item())
        score[i] = single_score
    return score


def mirex_score_single(true_key_index, predicted_key_index):
    true_key, predicted_key = (
        keys_idx_map[true_key_index],
        keys_idx_map[predicted_key_index],
    )

    if true_key == predicted_key:
        return 1

    if predicted_key in get_perfect_fifths(true_key):
        return 0.5

    if predicted_key == get_relative(true_key):
        return 0.3

    if predicted_key == get_parallel(true_key):
        return 0.2

    return 0


def get_perfect_fifths(key):
    key_number, key_type = get_key_parts(key)
    lower_fifth = key_number - 1
    higher_fifth = key_number + 1

    if lower_fifth < 1:
        lower_fifth = 12

    if higher_fifth > 12:
        higher_fifth = 1

    return f"{lower_fifth}{key_type}", f"{higher_fifth}{key_type}"


def get_key_parts(key):
    key_number = int(key[:-1])
    key_type = key[-1]

    return key_number, key_type


def get_relative(key):
    key_number, key_type = get_key_parts(key)

    if key_type == "B":
        return f"{key_number}A"

    return f"{key_number}B"


def get_parallel(key):
    scale = keys_map[key]
    scale_type = scale[-5:]
    new_scale_type = "minor"

    if scale_type == "minor":
        new_scale_type = "major"

    new_scale = scale[:-5] + new_scale_type

    return keys_map_inv[new_scale]
