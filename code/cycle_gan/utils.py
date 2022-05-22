import argparse
import cv2
import numpy as np
import torch
from augmentation import denormalize_image

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.0002) # 후반부 100 에폭은 learning rate decay가 적용이 되기 때문에 scheduler를 쓰든지 아님 만들든지 해야함.
    parser.add_argument("--decay_after", type=int, default=100)
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--last_epoch", type=int, default=100)
    parser.add_argument("--lr_decay_verbose", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cycle_loss_lambda", type=int, default=10)
    parser.add_argument("--data_root_A", type=str, default="code/cycle_gan/horse2zebra/trainA")
    parser.add_argument("--data_root_B", type=str, default="code/cycle_gan/horse2zebra/trainB")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--sample_interval", type=int, default=2500)
    parser.add_argument("--sample_save_dir", type=str, default='code/cycle_gan/results2/')

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