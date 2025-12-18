import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        z = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(
            z.unsqueeze(1),
            z.unsqueeze(0),
            dim=2
        )

        sim /= self.temperature

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -9e15)

        loss = F.cross_entropy(sim, labels)
        return loss
