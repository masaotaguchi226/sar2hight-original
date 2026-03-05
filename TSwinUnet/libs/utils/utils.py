import os
import torch
import random
import numpy as np

# ------------------------------------------------------------------------------
# 学習の初期化
def check_train_args(args):
    if not os.path.isdir(args.data_root):
        raise IOError(f'data_root {args.data_root} が存在しません')
    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} が存在しません')
    return

def init_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

# ------------------------------------------------------------------------------
# 推論の初期化
def check_predict_args(args):

    if not os.path.isdir(args.data_root):
        raise IOError(f'data_root {args.data_root} が存在しません')
    
    if not os.path.isdir(args.exp_root):
        raise IOError(f'exp_root {args.exp_root} が存在しません')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} が存在しません')

    return
