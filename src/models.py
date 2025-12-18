import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=False):
        super().__init__()

        if backbone == "resnet50":
            net = models.resnet50(pretrained=pretrained)
            self.out_channels = 2048
        elif backbone == "resnet18":
            net = models.resnet18(pretrained=pretrained)
            self.out_channels = 512
        else:
            raise ValueError("Unsupported backbone")

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x, return_feats=False):
        feats = {}

        x = self.relu(self.bn1(self.conv1(x)))
        feats["c1"] = x

        x = self.maxpool(x)
        x = self.layer1(x)
        feats["c2"] = x

        x = self.layer2(x)
        feats["c3"] = x

        x = self.layer3(x)
        feats["c4"] = x

        x = self.layer4(x)
        feats["c5"] = x

        if return_feats:
            return feats
        return feats["c5"]



class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SimCLRModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = ResNetEncoder(
            backbone="resnet50",
            pretrained=True
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # 🔥 MISSING LINE (FIX)
        self.projector = ProjectionHead(
            in_dim=self.encoder.out_channels,
            out_dim=proj_dim
        )

    def forward(self, x):
        h = self.encoder(x, return_feats=False)
        h = self.pool(h).flatten(1)
        return self.projector(h)



class SiameseChangeModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        return torch.abs(f1 - f2)


class ChangeDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return self.decoder(x)


class ChangeDetectionModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.siamese = SiameseChangeModel(encoder)
        self.decoder = ChangeDecoder(encoder.out_channels)

    def forward(self, x1, x2):
        f = self.siamese(x1, x2)
        return self.decoder(f)


class SiameseUNetChangeModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        self.up4 = UpBlock(2048, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x1, x2):
        f1 = self.encoder(x1, return_feats=True)
        f2 = self.encoder(x2, return_feats=True)


        d = {k: torch.abs(f1[k] - f2[k]) for k in f1}

        x = self.up4(d["c5"], d["c4"])
        x = self.up3(x, d["c3"])
        x = self.up2(x, d["c2"])
        x = self.up1(x, d["c1"])

        return self.out(x)
