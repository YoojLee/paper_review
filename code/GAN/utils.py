import argparse
import os
import torch

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-lr", default=0.0002, type=float) # argument 이름에 -나 -- 안붙여주면 무조건 들어가야 하는 인자로 인식함.
    arg_parser.add_argument("-n_epochs", default=200, type=int)
    arg_parser.add_argument("-batch_size", default=64, type=int)
    arg_parser.add_argument("-b1", default=0.5, type=float)
    arg_parser.add_argument("-b2", default=0.999, type=float)
    arg_parser.add_argument("-n_cpu", default=-1, type=int)
    arg_parser.add_argument("-latent_dim", default=100, type=int)
    arg_parser.add_argument("-img_size", default=28, type=int)
    arg_parser.add_argument("-channels", default=1, type=int)
    arg_parser.add_argument("-sample_interval", default=400, type=int)
    arg_parser.add_argument("-model_save_dir", default="./weights/")
    arg_parser.add_argument("--use_batch_norm", action="store_true")
    arg_parser.add_argument("--do_validation", action="store_true")
    args = arg_parser.parse_args()

    return args

def save_checkpoint(epoch, model_g, model_d, loss, optimizer_g, optimizer_d, save_dir, file_name):
    check_point = {
        'epoch': epoch,
        'generator': model_g.state_dict(),
        'discriminator': model_d.state_dict(),
        'loss': loss,
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict()
    }

    output_path = os.path.join(save_dir, file_name)
    torch.save(check_point, output_path)


if __name__ == "__main__":
    args = parse_args()
    print(args)