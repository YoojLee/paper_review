import argparse
from cmath import inf
import os
from re import I
from torchvision import transforms, datasets
from torchvision.utils import save_image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Generator, Discriminator
from utils import parse_args, save_checkpoint
from tqdm import tqdm



def validation(epoch, valid_loader, g, d, criterion, device, args):
    g.eval()
    d.eval()

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    with torch.no_grad():
        for step, (images, _) in pbar:
            b = images.shape[0]
            real_label = torch.ones(b).to(device)
            fake_label = torch.zeros(b).to(device)

            images = images.to(device)

            # for generator
            z = torch.randn(b, args.latent_dim).to(device)
            fake_images = g(z)

            g_loss = criterion(d(fake_images), real_label)

            # for discriminator
            real_loss = criterion(d(images), real_label)
            fake_loss = criterion(d(fake_images), fake_label)
            d_loss = (real_loss + fake_loss) / 2

            description = f"Validation #{epoch} || Generator Loss: {round(g_loss, 4)} || Discriminator Loss: {round(d_loss, 4)}"
            pbar.set_description(description)

        return (g_loss + d_loss) / 2
    

def train(train_loader, g, d, opt_g, opt_d, criterion, device, args, val_loader = None):
    best_loss = float(inf)

    for epoch in range(args.n_epochs):
        g.train()
        d.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (images, _) in pbar:

            # gradient 초기화
            opt_g.zero_grad()
            opt_d.zero_grad()
            
            # label 만들기
            b = images.shape[0]
            real_label = torch.ones(b).to(device)
            fake_label = torch.zeros(b).to(device)

            images = images.to(device) # dataloader는 device에 안 올려주고 대신 하나씩 꺼내올 때 device에 올려주기

            # train a generator
            z = torch.randn(b, args.latent_dim).to(device)
            fake_images = g(z)
            g_loss = criterion(d(fake_images), real_label) # input, target 순

            g_loss.backward()
            opt_g.step()

            # train a discriminator -> loss 구성이 real_loss와 fake_loss의 합으로 구성
            real_loss = criterion(d(images), real_label)
            fake_loss = criterion(d(fake_images.detach()), fake_label) # why detach? detach는 gradient를 끊어 놓는 것. fake_images에 detach를 해주지 않으면, generator에도 gradient가 흘러 들어갈 것 (in training discriminator)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            opt_d.step()


            description = f'Epoch: {epoch+1}/{args.n_epochs} || Step: {step+1}/{len(train_loader)} || Generator Loss: {round(g_loss, 4)} || Discriminator Loss: {round(d_loss, 4)}'
            pbar.set_description(description)

            batches_done = epoch * len(train_loader) + step

            if batches_done % args.sample_interval == 0:
                save_image(fake_images.detach()[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            
            if args.do_validation:
                val_loss = validation(epoch+1, val_loader, g, d, criterion, device, args)

                if val_loss < best_loss:
                    best_loss = val_loss

                    print(f"Best performance at epoch {epoch+1}")
                    print(f"Save model in {args.model_save_dir}")
                    
                    os.makedirs(args.model_save_dir, exist_ok=True)
                    save_checkpoint(epoch, g, d, best_loss, opt_g, opt_d, args.model_save_dir, file_name=f"gan_{best_loss}.pt")




def main(args):
    """
    load data, define loss and other things, train and evaluate

    - Args
        args: arguments parsed from the user command.    
    """
    # device 선언
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load the data
    os.makedirs("data/mnist", exist_ok=True) # exist_ok = True -> 기존에 존재하는 directory면 directory 만들지 않고 넘어가기

    train_loader = DataLoader(
        dataset = datasets.MNIST("data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])]
            )
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    
    # model 선언
    generator = Generator(latent_dim = args.latent_dim, output_size = (args.img_size, args.img_size, args.channels), normalize=args.use_batch_norm)
    discriminator = Discriminator(in_features = args.channels*args.img_size**2)
    
    # optimizer
    optimizer_G = torch.optim.Adam(params = generator.parameters(), lr=args.lr, betas = (args.b1, args.b2))
    optimizer_D = torch.optim.Adam(params = discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # criterion
    criterion = torch.nn.BCELoss()

    # train loop
    train(train_loader, generator, discriminator, optimizer_G, optimizer_D, criterion, device=device, args=args)



if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    args = parse_args()
    main(args)
    
