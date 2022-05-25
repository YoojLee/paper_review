"""
test.py
일단 모델 정의는 당연히 해야겠지? architecture instance는 다 선언하고 model load하기
"""
import os
from tqdm import tqdm

from augmentation import BaseAugmentation
from dataset import UnalignedDataset
from model import Generator
from utils import *

import torch
from torch.utils.data import DataLoader

def test(test_loader, G, F, device, save_dir):
    G.to(device)
    F.to(device)

    G.eval()
    F.eval()

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        for step, data in pbar:
            X, Y = data['A'], data['B']
            
            X, Y = X.to(device), Y.to(device)

            fake_Y = G(X)
            fake_X = F(Y)

            # make an image array
            XY = torch.cat([X, fake_Y, F(fake_Y)], dim = 3) # column-wise concat
            YX = torch.cat([Y, fake_X, G(fake_X)], dim = 3)

            os.makedirs(f"{save_dir}/XtoY/", exist_ok=True)
            os.makedirs(f"{save_dir}/YtoX/", exist_ok=True)
            
            # array로 저장해서 보는 게 빠를 것 같음. grid로 만들어서 보는 법 찾아볼 것.
            save_image(XY.clone().detach().cpu(), f"{save_dir}/XtoY/{step+1}.png")
            save_image(YX.clone().detach().cpu(), f"{save_dir}/YtoX/{step+1}.png")




if __name__ == "__main__":
    opt = parse_opt()

    fix_seed(opt.random_seed) # randomness 제어

    # device
    device = torch.device(f'cuda:{opt.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # define transforms
    transforms = BaseAugmentation()

    # load datasets
    test_data = UnalignedDataset(opt.data_root_A, opt.data_root_B, opt.is_train, transforms = transforms.transform)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=opt.is_train)

    # declare architectures
    G = Generator(init_channel=64, kernel_size=3, stride=2, n_blocks=9)
    F = Generator(init_channel=64, kernel_size=3, stride=2, n_blocks=9)

    G, F, _, _, _ = load_checkpoint(os.path.join(opt.last_checkpoint_dir, f"epoch{opt.load_epoch}.pth"), G, F)
    
    save_dir = os.path.join(opt.sample_save_dir, opt.exp_name, f"epoch{opt.load_epoch}")
    os.makedirs(save_dir, exist_ok=True)

    test(test_loader, G, F, device, save_dir)
