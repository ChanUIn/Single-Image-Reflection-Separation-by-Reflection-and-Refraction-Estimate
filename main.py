import os
import argparse
from solver import Solver
from data_loader import get_loader_test
from data_loader import get_loader_t
from data_loader import get_loader_val
from torch.backends import cudnn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    cudnn.benchmark = True

    if config.mode == 'train' or config.mode == 'val':
        data_loader_train = get_loader_t(config)
        # data_loader_val = get_loader_val(config)
    else:
        data_loader_train = None
        # data_loader_val = None

    data_loader_test = get_loader_test(config)
    data_loader_val = None
    # data_loader_test = None
    solver = Solver(data_loader_train, data_loader_val, data_loader_test, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'val':
        solver.val_all()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # parser 解析器
    # Model configuration. 模型配置
    parser.add_argument('--main_dir', type=str, default='D:/CHAN/absorption/main/matlab/dataset/training_data/')  # add argument 新增參數
    # Training configuration. iter 迭代器
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size') #default=4
    parser.add_argument('--num_epochs', type=int, default=200, help='number of total iterations for training D') # default=200
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--n_critic', type=int, default=5)
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=59, help='test model from this step')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--use_tensorboard', type=bool, default=True)
    parser.add_argument('--num_epoch_decay', type=int, default=50) #default=100

    parser.add_argument('--distance_type', type=int, default=0, help='1: de-rain, 0: reflection removal')
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    # Directories. 目錄
    parser.add_argument('--log_dir', type=str, default='D:/CHAN/absorption/main/ours/logs')
    parser.add_argument('--model_save_dir', type=str, default='D:/CHAN/absorption/main/ours/models')
    parser.add_argument('--sample_dir', type=str, default='D:/CHAN/absorption/main/ours/samples')
    parser.add_argument('--result_dir', type=str, default='D:/CHAN/absorption/main/ours/results')
    #parser.add_argument('__estimateR_dir', type=str, default='D:/CHAN/absorption/main/ours/estimateR/')  ##estimateR_dir
    #parser.add_argument('__out_channels', type=str, default='D:/CHAN/absorption/main/ours/results/estimateR/')  ###out_channels

    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--lr_update_step', type=int, default=1000)
    parser.add_argument('--log_step', type=int, default=10)
    config = parser.parse_args()
    print(config)
    main(config)

