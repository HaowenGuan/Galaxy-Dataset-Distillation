import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from reparam_module import ReparamModule


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette',
                        help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=4000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data.')
    parser.add_argument('--max_start_epoch', type=int, default=29,
                        help='max epoch we can start at. It must be smaller than buffer epoch.')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true',
                        help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    parser.add_argument('--max_files', type=int, default=None,
                        help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None,
                        help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--init_epoch', type=int, default=1, help="starting point of stage wise distillation")
    parser.add_argument('--fix_lr_teacher', action='store_true', help="Fix the lr_teach if you are confident")
    parser.add_argument('--prev_iter', type=int, default=1, help="Resume training start from previous iter")

    args = parser.parse_args()

    expert_dir = os.path.join(args.buffer_path, args.dataset, args.model)
    torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(0)))
    buffer = torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(0)))
    dp = np.zeros((30, 30))
    args.device = 'cuda'
    for x in range(10):
        print('doing expert', x)
        expert_trajectory = buffer[x]
        for i in range(30):
            starting_params = expert_trajectory[i]
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
            for j in range(i + 1, 30):
                target_params = expert_trajectory[j]
                target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
                diff = torch.square(starting_params - target_params)
                diff = torch.topk(diff, k=int(0.1 * diff.size(0))).values
                dp[i, j] += torch.sum(diff) * 10
                # dp[i, j] += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
    dp /= 10
    np.set_printoptions(linewidth=np.inf, suppress=True)
    print(np.around(dp, decimals=5))
    # main(args)
