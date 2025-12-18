import torch
import random

from src.data import ChangeDetectionDataset
from src.baseline import absolute_diff_baseline, morphological_cleanup


def compute_confusion(pred, gt):
    pred = pred.view(-1)
    gt = gt.view(-1)

    tp = torch.sum((pred == 1) & (gt == 1)).float()
    fp = torch.sum((pred == 1) & (gt == 0)).float()
    fn = torch.sum((pred == 0) & (gt == 1)).float()

    return tp, fp, fn


def iou_score(tp, fp, fn, eps=1e-6):
    return tp / (tp + fp + fn + eps)


def precision_score(tp, fp, eps=1e-6):
    return tp / (tp + fp + eps)


def recall_score(tp, fn, eps=1e-6):
    return tp / (tp + fn + eps)


def f1_score(tp, fp, fn, eps=1e-6):
    return (2 * tp) / (2 * tp + fp + fn + eps)


def evaluate_baseline(
    data_root,
    num_samples=50,
    threshold=0.2
):
    dataset = ChangeDetectionDataset(data_root)
    indices = random.sample(range(len(dataset)), num_samples)

    TP = FP = FN = 0.0

    for idx in indices:
        img1, img2, gt_mask = dataset[idx]

        _, pred_mask = absolute_diff_baseline(img1, img2, threshold)
        pred_mask = morphological_cleanup(pred_mask)

        tp, fp, fn = compute_confusion(pred_mask, gt_mask)

        TP += tp
        FP += fp
        FN += fn

    return {
        "IoU": iou_score(TP, FP, FN).item(),
        "Precision": precision_score(TP, FP).item(),
        "Recall": recall_score(TP, FN).item(),
        "F1": f1_score(TP, FP, FN).item()
    }
