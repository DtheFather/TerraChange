import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from src.transforms import SimCLRTransform
from src.models import SimCLRModel
from src.contrastive import NTXentLoss
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))



class SimCLRDataset(Dataset):
    def __init__(self, roots, transform):
        self.paths = []
        for root in roots:
            self.paths += glob.glob(os.path.join(root, "*.png"))
            self.paths += glob.glob(os.path.join(root, "*.jpg"))
        self.paths = sorted(self.paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        v1, v2 = self.transform(img)
        return v1, v2



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_roots = [
    os.path.join(BASE_DIR, "data/raw/levir/train/t1"),
    os.path.join(BASE_DIR, "data/raw/levir/train/t2")
    ]


    transform = SimCLRTransform(size=224)
    dataset = SimCLRDataset(data_roots, transform)
    
    print("Dataset size:", len(dataset))
    print("First 5 paths:", dataset.paths[:5])


    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model = SimCLRModel().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epochs = 15
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x1, x2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    os.makedirs("./outputs", exist_ok=True)
    torch.save(model.encoder.state_dict(), "./outputs/encoder_ssl_r50.pth")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("SimCLR Pretraining Loss")
    plt.savefig("./outputs/ssl_loss.png")
    plt.show()


if __name__ == "__main__":
    train()
