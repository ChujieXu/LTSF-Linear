import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')

# data loader
parser.add_argument('--data', type=str, default='Global_Wind_Temp', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# DLinear
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

Exp = Exp_Main
setting = 'Global_Wind_Temp'

if args.is_training:
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    torch.cuda.empty_cache()
else:
    exp = Exp(args)  # set experiments
    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.predict(setting, True)
    torch.cuda.empty_cache()
