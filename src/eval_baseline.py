import random
from pathlib import Path

from src.data import ChangeDetectionDataset
from src.baseline import absolute_diff_baseline, morphological_cleanup
from src.eval import (
    compute_confusion,
    iou_score,
    precision_score,
    recall_score,
    f1_score
)



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

