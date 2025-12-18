import os
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import urllib.request

from src.models import ResNetEncoder, SiameseUNetChangeModel


device = torch.device("cpu")

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


def download_if_missing(path, url):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
        return True
    return False


def load_change_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(base_dir, "outputs")

    encoder_path = os.path.join(outputs_dir, "encoder_ssl_r50.pth")
    change_path = os.path.join(outputs_dir, "change_model.pth")

    ENCODER_URL = "https://huggingface.co/DtheFather/terrachange-models/resolve/main/encoder_ssl_r50.pth"
    CHANGE_URL = "https://huggingface.co/DtheFather/terrachange-models/resolve/main/change_model.pth"

    downloaded_encoder = download_if_missing(encoder_path, ENCODER_URL)
    downloaded_change = download_if_missing(change_path, CHANGE_URL)

    encoder = ResNetEncoder(backbone="resnet50", pretrained=False)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    model = SiameseUNetChangeModel(encoder).to(device)
    model.load_state_dict(torch.load(change_path, map_location=device))

    model.eval()

    # 🔥 return model + whether any download happened
    return model, (downloaded_encoder or downloaded_change)



def preprocess(img):
    return transform(img).unsqueeze(0).to(device)


def predict(model, img1, img2, threshold=0.5):
    with torch.no_grad():
        logits = model(img1, img2)
        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float()
    return prob.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()


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
