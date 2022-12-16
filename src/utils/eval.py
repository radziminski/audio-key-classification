import re
import torch

from src.utils.mirex import mirex_score_single


def eval_on_full_songs(datamodule, model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    datamodule.prepare_data()
    datamodule.setup(stage="test")

    losses_sum = 0
    losses_count = 0

    for dataset in datamodule.instantiated_datasets:
        losses_sum += test_dataset(dataset, model)
        losses_count += 1

    print(f"MIREX score for whole test dataset: {losses_sum / losses_count}")


def test_dataset(dataset, model):
    previous_song_num = None
    current_batch_samples = []
    current_batch_labels = []
    losses_sum = 0
    losses_count = 0
    for index, sample in enumerate(dataset):
        image = sample[0]
        label = sample[1]
        full_sample_path = dataset.samples[index][0]
        filepart = extract_filepart(full_sample_path)
        song_num = extract_song_num(filepart)

        if previous_song_num is None:
            previous_song_num = song_num
            current_batch_samples.append(image)
            current_batch_labels.append(label)

        if song_num is previous_song_num:
            current_batch_samples.append(image)
            current_batch_labels.append(label)
        else:
            batch = torch.stack(current_batch_samples, dim=0), current_batch_labels
            loss, _, _ = test_step(model, batch)
            losses_sum += loss
            losses_count += 1
            current_batch_samples = [image]
            current_batch_labels = [label]
            previous_song_num = song_num
    print(f"Test dataset at {dataset.root}")
    print(f"MIREX score: {losses_sum / losses_count}")
    return losses_sum / losses_count


def extract_song_num(filename: str):
    match = re.match(r"^[^-]*", filename)
    return int(match.group(0))


def extract_filepart(full_path: str):
    match = re.search(r"^.*/([^/]*)$", full_path)
    return match.group(1)


def test_step(model, batch):
    x, y = batch
    logits = model.forward(x)
    logits_sum = torch.sum(logits, dim=0)
    prediction = torch.argmax(logits_sum, dim=0)
    loss = mirex_score_single(prediction.item(), y[0])
    return loss, prediction, y
