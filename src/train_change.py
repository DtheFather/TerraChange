import torch
import torch.nn.functional as F
from src.models import ResNetEncoder, SiameseUNetChangeModel
from src.losses import ChangeDetectionLoss
from src.data import ChangeDetectionDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


encoder = ResNetEncoder(backbone="resnet50", pretrained=False)
encoder.load_state_dict(
    torch.load("outputs/encoder_ssl_r50.pth", map_location=device)
)
 

for p in encoder.parameters():
    p.requires_grad = False


model = SiameseUNetChangeModel(encoder).to(device)


dataset = ChangeDetectionDataset(
    root_dir="data/raw/levir/train",
    image_size=256
)

criterion = ChangeDetectionLoss()


loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

model.train()

for epoch in range(40):
    total_loss = 0
    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        pred = model(x1, x2)

        if pred.shape[-2:] != y.shape[-2:]:
            pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss {total_loss/len(loader):.4f}")


torch.save(model.state_dict(), "outputs/change_model.pth")

