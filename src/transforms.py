import random
from PIL import ImageFilter
import torchvision.transforms as T


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class SimCLRTransform:
    def __init__(self, size=224):
        color_jitter = T.ColorJitter(0.8, 0.8, 0.8, 0.2)

        self.transform = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur()], p=0.5),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, img):
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2
