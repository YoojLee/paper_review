import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=.0002) # 후반부 100 에폭은 learning rate decay가 적용이 되기 때문에 scheduler를 쓰든지 아님 만들든지 해야함.
    parser.add_argument("--decay_after", type=int, default=100)
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cycle_loss_lambda", type=int, default=10)
    parser.add_argument("--data_root_A", type=str, default="code/cycle_gan/horse2zebra/trainA")
    parser.add_argument("--data_root_B", type=str, default="code/cycle_gan/horse2zebra/trainB")
    parser.add_argument("--is_train", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--sample_interval", type=int, default=5000)
    parser.add_argument("--sample_save_dir", type=str, default='code/cycle_gan/results/')

    opt = parser.parse_args()

    return opt