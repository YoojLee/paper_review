import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


class BaseAugmentation:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]):
        self.transform = A.Compose([
                A.Normalize(mean=mean, std=std, always_apply=True),
                ToTensorV2()
        ])

def denormalize_image(image, mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]):
    """
    Returns Denormalized images

    - Args
        image (torch.Tensor): an image tensor with (N,C,H,W) or (C,H,W)
        mean (List[float]): a list of mean values of each channels used in a normalization (ImageNet)
        std (List[float]): a list of standard deviations used in a normalization (ImageNet)
    """
    
    mean = 255.0*np.array(mean).reshape(-1,1,1)
    std = 255.0*np.array(std).reshape(-1,1,1)

    if len(image.shape) == 4 and image.shape[0]==1:
        image = image.squeeze()
    
    denorm_image = np.clip(image*std+mean, 0, 255)

    return denorm_image