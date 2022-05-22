import argparse
import cv2
import numpy as np
import random
import torch
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
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.0002) # 후반부 100 에폭은 learning rate decay가 적용이 되기 때문에 scheduler를 쓰든지 아님 만들든지 해야함.
    parser.add_argument("--beta1", type=float, default=.5, help = "beta1 for Adam optimizer")
    parser.add_argument("--decay_after", type=int, default=100)
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--last_epoch", type=int, default=100)
    parser.add_argument("--lr_decay_verbose", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cycle_loss_lambda", type=float, default=10.0)
    parser.add_argument("--idt_loss_lambda", type=float, default=0.5)
    parser.add_argument("--data_root_A", type=str, default="code/cycle_gan/horse2zebra/trainA")
    parser.add_argument("--data_root_B", type=str, default="code/cycle_gan/horse2zebra/trainB")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=16)
    
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=0)

    # wandb & logging
    parser.add_argument("--prj_name", type=str, default="cycle_gan")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--logging_interval", type=int, default=25)
    parser.add_argument("--sample_save_dir", type=str, default='code/cycle_gan/results/')

    opt = parser.parse_args()

    return opt

def save_image(image, save_path, denormalize=True):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    if denormalize:
        image = denormalize_image(image)
    
    image = image.astype(np.uint8).copy()

    cv2.imwrite(save_path, cv2.cvtColor(image.transpose(1,2,0), cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    import pickle

    with open("/home/workspace/code/cycle_gan/results2/example.pickle", "rb") as f:
        sample = pickle.load(f)

    print(sample.min(), sample.max())
    sample = denormalize_image(sample)
    sample = sample.astype(np.uint8)
    print(sample.shape)

    cv2.imwrite("/home/workspace/code/cycle_gan/results2/example.png", sample.transpose(1,2,0))
    cv2.imwrite("/home/workspace/code/cycle_gan/results2/example_convert.png", cv2.cvtColor(sample.transpose(1,2,0), cv2.COLOR_BGR2RGB))