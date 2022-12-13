NOTES_LIST = ["C", "D", "E", "F", "G", "A", "B"]


def map_scale_to_bemol(scale):
    if not "♯" in scale and not "#" in scale:
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
