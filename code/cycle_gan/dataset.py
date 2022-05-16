"""
데이터 불러오는 형식은 기존과 크게 다르지 않음.
__getitem__을 어떻게 구현할 거냐의 문제인데 그냥 A,B를 둘 다 리턴하는 형식임. 실제 코드에서는 path도 같이 넘겨주는데 왜 path도 같이 넘겨주는지는 모르겠음.

init할 때는 그냥 image directory만 받아오고 getitem 내에서 이미지를 직접 읽어오는 형태로 구현되어 있음.
"""
import os
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict


class UnalignedDataset(Dataset):
    """
    원래 reference 코드 같은 경우에는 BaseDataset이라는 datasets를 위한 추상 클래스를 하나 정의해서 그걸 상속받게 되어 있음.
    -> 왜 굳이 torch 내의 Dataset을 상속받게 하지 않고 BaseDataset을 하나 더 만든지는 모르겠음.
    내가 생각할 때는 modify_commandline_options라는 정적메소드를 계속 구현하지 않고 쓰게하려는 거 같기도..?

    하지만, 여기서는 그냥 Datasets를 상속받아서 사용하도록 함. 일단은 A to B만 구현한다. 양방향은 나중에 구현하자.
    """

    def __init__(self, data_root_A: str, data_root_B: str, is_train:bool=True, transforms=ToTensorV2()):
        """
        - Args
            data_root_A (str): a data root for source Domain A
            data_root_B (str): a data root for target Domain B
        """
        super(UnalignedDataset, self).__init__()
        self.data_root_A = data_root_A
        self.data_root_B = data_root_B
        self.is_train = is_train
        self.transforms = transforms

        paths_A = sorted(self._load_image_path(self.data_root_A))
        paths_B = sorted(self._load_image_path(self.data_root_B))

        # A와 B의 길이가 다른 경우를 위해 handling
        self.image_paths_A, self.image_paths_B = self._adjust_dataset_length(paths_A, paths_B)
        

    def __len__(self):
        """
        - Returns
            a length of smaller datasets between A and B
        """
        return len(self.image_paths_A)
        
    def __getitem__(self, index:int)->Dict:
        """
        - Args
            index (int)
        - Returns
            dct: A dictionary with keys ('A', 'B') and items (image array of A, image array of B)
        """
        A = cv2.cvtColor(cv2.imread(self.image_paths_A[index]), cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(cv2.imread(self.image_paths_B[index]), cv2.COLOR_BGR2RGB)

        if self.transforms:
            A = self.transforms(image=A)['image'] # for albumentations
            B = self.transforms(image=B)['image']
        
        return {'A': A, 'B': B}
        
    def _load_image_path(self, data_dir:str)->List[str]:
        """
        - Args
            data_dir (str): a directory where image is stored. must be given as an absolute path.
        """
        image_path  = glob.glob(data_dir+"/*") # 상위 경로까지 포함해서 리턴
        
        return image_path

    def _adjust_dataset_length(self, paths_A:str, paths_B:str):
        min_len = min(len(paths_A), len(paths_B))
        return paths_A[:min_len], paths_B[:min_len]


if __name__ == "__main__":
    img_dir = "/home/workspace/code/cycle_gan/horse2zebra/trainA/"
    img_dir = glob.glob(img_dir+"/*")
    img = cv2.imread(img_dir[2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape) # 이미지가 다 scale되어있음을 확인.