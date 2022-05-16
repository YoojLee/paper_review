import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from augmentation import BaseAugmentation
from criterion import *
from dataset import UnalignedDataset
from model import *
from utils import parse_opt

from tqdm import tqdm
from functools import partial
import itertools
import os

def train(train_loader, n_epochs, models, optimizers, lambda_cyc, device, sample_interval, sample_save_dir):
    
    os.makedirs(sample_save_dir, exist_ok=True)

    G, F, D_x, D_y = models
    optim_G, optim_D = optimizers

    criterion_G = AdversarialLoss(mode='g') # 왜 direction끼리 얘를 공유해야하는지 잘 모르겠음..
    criterion_D = AdversarialLoss(mode='d')
    criterion_cyc = nn.L1Loss()

    G.train()
    F.train()
    D_x.train()
    D_y.train()

    for epoch in range(n_epochs):
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, data in pbar:
            # 매 step마다 gradient 초기화
            optim_G.zero_grad()
            optim_D.zero_grad()

            # label 만들기
            X, Y = data['A'], data['B']
            b = X.shape[0]

            X, Y = X.to(device), Y.to(device)

            #### Generator ####
            fake_xy = G(X)
            fake_yx = F(Y)

            d_xy = D_y(fake_xy) # D_target
            d_yx = D_x(fake_yx)

            real_label = torch.tensor([1.0]).expand_as(d_xy).to(device)
            fake_label = torch.tensor([0.0]).expand_as(d_yx).to(device)

            # adversarial loss 계산
            loss_G_xy = criterion_G.forward_G(d_xy, real_label)
            loss_F_yx = criterion_G.forward_G(d_yx, real_label)

            # cycle loss 계산
            loss_cyc_forward = criterion_cyc(F(fake_xy), Y)
            loss_cyc_backward = criterion_cyc(G(fake_yx), X)

            loss_G = loss_G_xy + loss_F_yx + lambda_cyc*(loss_cyc_forward+loss_cyc_backward)
            
            loss_G.backward()
            optim_G.step()

            #### Discriminator ####
            loss_D_xy = criterion_D.forward_D(D_y(Y), real_label, d_xy, fake_label)
            loss_D_yx = criterion_D.forward_D(D_x(X), real_label, d_yx, fake_label)

            loss_D_xy.backward()
            loss_D_yx.backward()

            optim_D.step()

            description = f'Epoch: {epoch+1}/{n_epochs} || Step: {step+1}/{len(train_loader)} || Generator Loss: {round(loss_G.item(), 4)} || Discriminator Loss (XY, YX): {round(loss_D_xy.item(), 4)},{round(loss_D_yx.item(), 4)}'
            pbar.set_description(description)

            batches_done = epoch * len(train_loader) + step

            if batches_done % sample_interval == 0:
                save_image(fake_xy.clone().detach()[:25], f"{sample_save_dir}/%d.png" % batches_done, nrow=5, normalize=True)

    

def main():
    opt = parse_opt()

    # device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #device='cpu'

    # define transforms
    transforms = BaseAugmentation()

    # load datasets
    train_data = UnalignedDataset(opt.data_root_A, opt.data_root_B, opt.is_train, transforms = transforms.transform)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)

    # load model
    G = Generator(init_channel=64, kernel_size=3, stride=2, n_blocks=9).to(device)
    F = Generator(init_channel=64, kernel_size=3, stride=2, n_blocks=9).to(device)
    D_x = Discriminator().to(device)
    D_y = Discriminator().to(device)

    # define optimizer
    optimizer = partial(torch.optim.Adam, lr=opt.lr)

    optim_G = optimizer(params = itertools.chain(G.parameters(), F.parameters()))
    #optim_F = optimizer(F.parameters())
    optim_D = optimizer(params = itertools.chain(D_x.parameters(), D_y.parameters()))
    #optim_Dy = optimizer(D_y.parameters())


    kwargs = {
        'train_loader': train_loader,
        'n_epochs': opt.n_epochs,
        'models': [G, F, D_x, D_y],
        'optimizers': [optim_G, optim_D],
        'lambda_cyc': opt.cycle_loss_lambda,
        'sample_interval': opt.sample_interval,
        'sample_save_dir': opt.sample_save_dir,
        'device': device
    }

    train(**kwargs)


if __name__ == "__main__":
    main()
    