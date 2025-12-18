import os
import random
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import ChangeDetectionDataset
from src.baseline import absolute_diff_baseline, morphological_cleanup


def run_baseline_visualization(
    data_root,
    output_dir="outputs",
    num_samples=5,
    threshold=0.2
):
    os.makedirs(output_dir, exist_ok=True)

    dataset = ChangeDetectionDataset(data_root)

    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        img1, img2, gt_mask = dataset[idx]

        diff, pred_mask = absolute_diff_baseline(img1, img2, threshold)
        pred_mask = morphological_cleanup(pred_mask)

        fig = plt.figure(figsize=(12, 3))

        plt.subplot(1, 4, 1)
        plt.imshow(img1.permute(1, 2, 0))
        plt.title("T1")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(img2.permute(1, 2, 0))
        plt.title("T2")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(diff.squeeze(), cmap="hot")
        plt.title("|T1 - T2|")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(pred_mask.squeeze(), cmap="gray")
        plt.title("Predicted Change")
        plt.axis("off")

        save_path = Path(output_dir) / f"baseline_sample_{i+1}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")
