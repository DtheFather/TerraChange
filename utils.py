import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from src.models import ResNetEncoder, ChangeDetectionModel, SiameseUNetChangeModel


device = torch.device("cpu")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def load_change_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "outputs", "change_model.pth")

    encoder = ResNetEncoder(
        backbone="resnet50",
        pretrained=False
    )

    model = SiameseUNetChangeModel(encoder).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model

def preprocess(img):
    return transform(img).unsqueeze(0).to(device)

def predict(model, img1, img2, threshold=0.5):
    with torch.no_grad():
        logits = model(img1, img2)
        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float()
    return prob.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()

import numpy as np
import cv2

def compute_metrics(pred, gt, eps=1e-6):
    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred.astype(np.float32),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return iou, precision, recall, f1
