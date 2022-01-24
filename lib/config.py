from .yacs import CfgNode as CN
import argparse
import os
import torch
import numpy as np

cfg = CN()

cfg.K = 2
cfg.d = 2
cfg.M = 2
cfg.N = 8
cfg.M_RF = 2
cfg.N_RF = 4
cfg.sigma = 1
cfg.epsilon = 1e-5
cfg.eta = 1e-3
cfg.max_iteration = 500
cfg.average_number = 100
cfg.power = 100
cfg.D_in = 10
cfg.D_out = 7
cfg.USE_GPU = False

def make_cfg(args):
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])
    cfg.local_rank = args.local_rank
    return cfg

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="lib/default.yaml", type=str)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)