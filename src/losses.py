import torch
import torch.nn.functional as F
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()

def bce_dice_loss(pred, target):
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    dice = dice_loss(pred, target)
    return bce + dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class ChangeDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, targets):
        return 0.7 * self.dice(logits, targets) + \
               0.3 * self.focal(logits, targets)
