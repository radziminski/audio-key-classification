import torch
import torch.nn.functional as F
from .mirex import (
    keys_idx_map,
    keys_idx_map_inv,
    get_perfect_fifths,
    get_relative,
    get_parallel,
)

CORRECT_WEIGHT = 1
PERFECT_FIFTH_WEIGHT = 0.5
RELATIVE_WEIGHT = 0.3
PARALLEL_WEIGHT = 0.2
INCORRECT_WEIGHT = 0


def get_mirex_classes_weights(y_true):
    batch_size = len(y_true)
    mirex_weights = torch.zeros(batch_size, 24) + INCORRECT_WEIGHT

    for index, y_true_single in enumerate(y_true):
        y_true_index = int(y_true_single)
        mirex_weights[index][y_true_index] = CORRECT_WEIGHT

        true_key = keys_idx_map[y_true_index]
        perfect_first, perfect_second = get_perfect_fifths(true_key)
        perfect_first_index = keys_idx_map_inv[perfect_first]
        perfect_second_index = keys_idx_map_inv[perfect_second]

        mirex_weights[index][perfect_first_index] = PERFECT_FIFTH_WEIGHT
        mirex_weights[index][perfect_second_index] = PERFECT_FIFTH_WEIGHT

        relative_index = keys_idx_map_inv[get_relative(true_key)]
        parallel_index = keys_idx_map_inv[get_parallel(true_key)]

        mirex_weights[index][relative_index] = RELATIVE_WEIGHT
        mirex_weights[index][parallel_index] = PARALLEL_WEIGHT

    return mirex_weights


def mirex_loss_v1(y_predictions, y_true, device="cuda"):
    mirex_weights = get_mirex_classes_weights(y_true).to(device)

    x_soft_maxed = torch.nn.functional.softmax(y_predictions, dim=1)
    loss_per_logit = F.binary_cross_entropy_with_logits(
        x_soft_maxed, mirex_weights, reduction="none"
    )

    return loss_per_logit.mean()


def mirex_loss_v2(y_predictions, y_true, device="cuda"):
    mirex_weights = get_mirex_classes_weights(y_true).to(device)
    mirex_weights[mirex_weights == 0] = 1
    mirex_weights[mirex_weights != 1] = 1 - mirex_weights[mirex_weights != 1]

    y_true_one_hot_encoded = torch.zeros((len(y_true), 24), device=device)
    y_true_one_hot_encoded[y_true] = 1

    x_soft_maxed = torch.nn.functional.softmax(y_predictions, dim=1)
    loss_per_logit = F.binary_cross_entropy_with_logits(
        x_soft_maxed, y_true, reduction="none"
    )

    return (loss_per_logit[:] * mirex_weights).mean()
