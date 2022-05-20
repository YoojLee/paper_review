import torch
from torch.utils.data import DataLoader

from augmentation import BaseAugmentation
from criterion import *
from dataset import UnalignedDataset
from model import *
from utils import *
from scheduler import DelayedLinearDecayLR

import cv2
import numpy as np
import pickle
from tqdm import tqdm
from functools import partial
import itertools
import os

    

def train(train_loader, n_epochs, models, optimizers, schedulers, lambda_cyc, device, sample_interval, sample_save_dir):
    
    os.makedirs(sample_save_dir, exist_ok=True)

    G, F, D_x, D_y = models
    optim_G, optim_D = optimizers
    scheduler_G, scheduler_D = schedulers

    criterion_G = AdversarialLoss() # 왜 direction끼리 얘를 공유해야하는지 잘 모르겠음..
    criterion_D = AdversarialLoss()
    criterion_cyc = CycleConsistencyLoss()

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

            X, Y = X.to(device), Y.to(device)

            #### Generator ####

            for p_x, p_y in zip(D_x.parameters(), D_y.parameters()):
                p_x.requires_grad = False
                p_y.requires_grad = False

            g_x = G(X)
            f_y = F(Y)

            d_g_x = D_y(g_x) # D_target
            d_f_y = D_x(f_y)

            real_label = torch.tensor([1.0]).expand_as(d_g_x).to(device)
            fake_label = torch.tensor([0.0]).expand_as(d_f_y).to(device)

            # adversarial loss 계산 -> 가짜를 진짜로 판별할 확률 최대화
            loss_G_xy = criterion_G.forward_G(d_g_x, real_label)
            loss_F_yx = criterion_G.forward_G(d_f_y, real_label)

            # cycle loss 계산
            loss_cyc = criterion_cyc(X, Y, F(g_x), G(f_y))

            loss_G = loss_G_xy + loss_F_yx + lambda_cyc*loss_cyc
            
            loss_G.backward()
            optim_G.step() # alternating training 해야돼서 G랑 D는 optimizer 따로 쓰는 거임.

            #### Discriminator ####
            for p_x, p_y in zip(D_x.parameters(), D_y.parameters()):
                p_x.requires_grad = True
                p_y.requires_grad = True

            loss_D_xy = criterion_D.forward_D(D_y(Y), real_label, d_g_x, fake_label)
            loss_D_yx = criterion_D.forward_D(D_x(X), real_label, d_f_y, fake_label)

            loss_D_xy.backward() # loss_G backward는 왜 퉁쳐서 하면서 얘는 따로 함..?
            loss_D_yx.backward()

            optim_D.step()

            description = f'Epoch: {epoch+1}/{n_epochs} || Step: {step+1}/{len(train_loader)} || Generator Loss: {round(loss_G.item(), 4)} || Discriminator Loss (XY, YX): {round(loss_D_xy.item(), 4)},{round(loss_D_yx.item(), 4)}'
            pbar.set_description(description)

            # batches_done = epoch * len(train_loader) + step

            # if batches_done % sample_interval == 0:
            #     save_image(g_x.clone().detach(), f"{sample_save_dir}/%d.png" % batches_done, nrow=5, normalize=True)
        scheduler_G.step()
        scheduler_D.step()

        # fake_np = g_x.clone().detach().cpu().numpy()
        # with open(f"{sample_save_dir}/example.pickle", "wb") as f:
        #     pickle.dump(fake_np, f)

        save_image(g_x.clone().detach().cpu(), f"{sample_save_dir}/epoch{epoch+1}.png")
    

def main():
    opt = parse_opt()

    # device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')    

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
    optim_D = optimizer(params = itertools.chain(D_x.parameters(), D_y.parameters()))

    # scheduler
    
    scheduler_G = DelayedLinearDecayLR(optim_G, opt.lr, opt.target_lr, opt.last_epoch, decay_after=opt.decay_after, verbose=opt.lr_decay_verbose)
    scheduler_D = DelayedLinearDecayLR(optim_D, opt.lr, opt.target_lr, opt.last_epoch, decay_after=opt.decay_after, verbose=opt.lr_decay_verbose)


    kwargs = {
        'train_loader': train_loader,
        'n_epochs': opt.n_epochs,
        'models': [G, F, D_x, D_y],
        'optimizers': [optim_G, optim_D],
        'schedulers': [scheduler_G, scheduler_D],
        'lambda_cyc': opt.cycle_loss_lambda,
        'sample_interval': opt.sample_interval,
        'sample_save_dir': opt.sample_save_dir,
        'device': device
    }

    train(**kwargs)


if __name__ == "__main__":
    main()
    