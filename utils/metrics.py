"""
Evaluation metrics for action recognition downstream tasks.

Provides top-1/top-5 accuracy, mean class accuracy, and per-class accuracy
for UCF101, HMDB51, and Something-Something V2 (SSv2) benchmarks.

Developed by Elijah from Massey University.
"""

import torch
import numpy as np


def compute_top1_accuracy(logits, labels):
    """Compute top-1 accuracy.

    Args:
        logits: (B, num_classes) raw logits or softmax probabilities.
        labels: (B,) ground-truth class indices.

    Returns:
        Scalar tensor with top-1 accuracy in [0, 1].
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean()


def compute_top5_accuracy(logits, labels):
    """Compute top-5 accuracy.

    Args:
        logits: (B, num_classes) raw logits.
        labels: (B,) ground-truth class indices.

    Returns:
        Scalar tensor with top-5 accuracy in [0, 1].
    """
    num_classes = logits.size(1)
    k = min(5, num_classes)
    _, topk_preds = logits.topk(k, dim=1)
    correct = topk_preds.eq(labels.unsqueeze(1).expand_as(topk_preds))
    return correct.any(dim=1).float().mean()


def compute_mean_class_accuracy(all_preds, all_labels, num_classes):
    """Compute mean per-class accuracy (macro-average over classes).

    Args:
        all_preds: (N,) predicted class indices for entire test set.
        all_labels: (N,) ground-truth class indices.
        num_classes: Total number of classes.

    Returns:
        Float scalar — mean class accuracy.
    """
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    for c in range(num_classes):
        mask = all_labels == c
        per_class_total[c] = mask.sum()
        if mask.sum() > 0:
            per_class_correct[c] = (all_preds[mask] == c).sum().float()

    valid = per_class_total > 0
    if valid.sum() == 0:
        return 0.0
    return (per_class_correct[valid] / per_class_total[valid]).mean().item()


def compute_per_class_accuracy(all_preds, all_labels, num_classes):
    """Compute accuracy for each individual class.

    Args:
        all_preds: (N,) predicted class indices.
        all_labels: (N,) ground-truth class indices.
        num_classes: Total number of classes.

    Returns:
        Dict mapping class index -> accuracy (float). Classes with no
        samples are omitted.
    """
    result = {}
    for c in range(num_classes):
        mask = all_labels == c
        total = mask.sum().item()
        if total > 0:
            correct = (all_preds[mask] == c).sum().item()
            result[c] = correct / total
    return result
