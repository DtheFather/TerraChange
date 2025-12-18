import torch
import torch.nn.functional as F

def absolute_diff_baseline(img1, img2, threshold=0.2):
    """
    img1, img2: torch tensors [3, H, W] in [0,1]
    returns: binary mask [1, H, W]
    """

    diff = torch.mean(torch.abs(img1 - img2), dim=0, keepdim=True)

    mask = (diff > threshold).float()

    return diff, mask


def morphological_cleanup(mask, kernel_size=5):
    """
    Simple morphology using max-pooling
    mask: [1, H, W]
    """

    mask = mask.unsqueeze(0)
    cleaned = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
    cleaned = F.max_pool2d(cleaned, kernel_size, stride=1, padding=kernel_size // 2)

    return cleaned.squeeze(0)
