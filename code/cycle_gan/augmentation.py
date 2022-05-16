import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class BaseAugmentation:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
        ])