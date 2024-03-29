import argparse
import cv2
import numpy as np
import os
import random
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional

from augmentation import denormalize_image

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # miscellaneous arguments for training
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=24)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=0)

    # image datasets
    parser.add_argument("--data_root_A", type=str, default="code/cycle_gan/horse2zebra/trainA")
    parser.add_argument("--data_root_B", type=str, default="code/cycle_gan/horse2zebra/trainB")

    # training details
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--beta1", type=float, default=.5, help = "beta1 for Adam optimizer")
    parser.add_argument("--lr", type=float, default=.0002)

    # learning rate decay
    parser.add_argument("--decay_after", type=int, default=100)
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--total_iters", type=int, default=100)
    parser.add_argument("--lr_decay_verbose", type=bool, default=False)

    # hyperparameter in respect to a loss
    parser.add_argument("--lambda_cyc", type=float, default=10.0)
    parser.add_argument("--lambda_idt", type=float, default=0.5)
    
    # wandb & logging
    parser.add_argument("--prj_name", type=str, default="cycle_gan")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--sample_save_dir", type=str, default='code/cycle_gan/test_results/') # only for test.py
    parser.add_argument("--last_checkpoint_dir", type=str, default="code/cycle_gan/weights/D_averaging")
    parser.add_argument("--checkpoint_dir", type=str, default="code/cycle_gan/weights")
    parser.add_argument("--load_epoch", type=int, default=150)
    
    # resume from
    parser.add_argument("--resume_from", action="store_true")

    opt = parser.parse_args()

    return opt

def save_image(image:torch.Tensor, save_path:str, denormalize:bool=True):
    """
    convert a torch.Tensor to a numpy array-> denormalize (if needed) -> type casting -> writing image to the given path
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    if denormalize:
        image = denormalize_image(image)
    
    image = image.astype(np.uint8).copy()

    cv2.imwrite(save_path, cv2.cvtColor(image.transpose(1,2,0), cv2.COLOR_BGR2RGB))

def save_checkpoint(epoch, G, F, D_x, D_y, optim_G, optim_D, scheduler_G, scheduler_D, saved_dir, file_name):
    check_point = {'epoch': epoch,
                    'G': G.state_dict(),
                    'F': F.state_dict(),
                    'D_x': D_x.state_dict(),
                    'D_y': D_y.state_dict(),
                    'optimG_state_dict': optim_G.state_dict(),
                    'optimD_state_dict': optim_D.state_dict()
                    }
    if scheduler_G and scheduler_D:
        check_point['scheduler_G_state_dict'] = scheduler_G.state_dict()
        check_point['scheduler_D_state_dict'] = scheduler_D.state_dict()

    os.makedirs(saved_dir, exist_ok=True) # make a directory to save a model if not exist.

    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)


def load_checkpoint(checkpoint_path, G, F, D_x:Optional[torch.nn.Module]=None, D_y:Optional[torch.nn.Module]=None, \
                    optim_G:Optional[Optimizer]=None, optim_D:Optional[Optimizer]=None, \
                    scheduler_G:Optional[_LRScheduler]=None, scheduler_D:Optional[_LRScheduler]=None, mode:str="model"):
    """
    - Args
        checkpoint_path (str): a path of the saved checkpoint
        G (network): generator (X->Y)
        F (network): generator (Y->X)
        D_x (network, Optional): discriminator (for X)
        D_y (network, Optional): discriminator (for Y)
        optim_G (Optional[Optimizer]): optimizer for generator
        optim_D (Optional[Optimizer]): optimizer for discriminator
        scheduler_G (Optional[_LRScheduler]): scheduler for Generator
        scheduler_D (Optional[_LRScheduler]): scheduler for Discriminator
        mode (str): "model" for model only or "all" for all the instances.
    """
    # load model if resume_from is set
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['G'])
    F.load_state_dict(checkpoint['F'])

    if D_x and D_y:
        D_x.load_state_dict(checkpoint['D_x'])
        D_y.load_state_dict(checkpoint['D_y'])

    start_epoch = checkpoint['epoch']

    if mode == "model":
        return G, F, D_x, D_y, start_epoch

    if mode =="all":
        optim_G.load_state_dict(checkpoint['optimG_state_dict'])
        optim_D.load_state_dict(checkpoint['optimD_state_dict'])
        
        if scheduler_G and scheduler_D:
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

            return G, F, D_x, D_y, optim_G, optim_D, scheduler_G, scheduler_D, start_epoch
        
        return G, F, D_x, D_y, optim_G, optim_D, start_epoch
    
    else:
        raise ValueError("mode should be one of 'model' or 'all'")