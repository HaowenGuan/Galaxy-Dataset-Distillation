import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import copy
import random
from reparam_module import ReparamModule
from torchvision.utils import save_image
from astropy.io import fits
import cv2 as cv
from torchvision import datasets, transforms
from gzoo2_dataset import GZooDataset, CustomDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    mean_acc_all = []
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=2)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    image_syn = torch.load("/data/sbcaesar/mac_galaxy/logged_files/GZoo2_aug/Final-GZoo2-1ipc-aug-independent-lr/images_last.pt")

    if args.flip_h or args.flip_v or args.transpose:
        image_syn_list = [image_syn]
        label_syn_list = [label_syn]

        if args.flip_h:
            print("Flipping images horizontally for augmentation")
            image_syn_list.append(torch.flip(image_syn, [3]))
            label_syn_list.append(label_syn)


        if args.flip_v:
            print("Flipping images vertically for augmentation")
            image_syn_list.append(torch.flip(image_syn, [2]))
            label_syn_list.append(label_syn)

        if args.transpose:
            print("Transposing images for augmentation")
            image_syn_list.append(torch.transpose(image_syn, 2, 3))
            label_syn_list.append(label_syn)

        image_syn = torch.cat(image_syn_list, dim=0)
        label_syn = torch.cat(label_syn_list, dim=0)

    if args.rotate:
        print("Rotating images for augmentation")
        image_syn_list = [image_syn]
        for i in range(3):
            image_syn_list.append(torch.rot90(image_syn, i+1, [2, 3]))
        image_syn = torch.cat(image_syn_list, dim=0)
        label_syn = torch.cat([label_syn for _ in range(4)], dim=0)


    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    for i in range(1, 2):
        args.lr_net = [0.0007435352890752256, 0.0005036790971644223, 0.00040825619362294674, 0.0003518034936860204, 0.0003059091977775097, 0.00027298444183543324, 0.00023740495089441538, 0.00020283406774979085, 0.00019477325258776546, 0.00017433073662687093]
        # mean_v = sum(args.lr_net) / len(args.lr_net)
        # args.lr_net = [mean_v] * 10
        # for j in range(20):
        #     args.lr_net.append(args.lr_net[-1] * 0.8)

        for model_eval in model_eval_pool:
            accs_test = []
            accs_train = []
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                eval_labs = label_syn
                with torch.no_grad():
                    image_save = image_syn
                image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification


                _, acc_train, acc_test, train_cf, test_cf = evaluate_synset(1, it_eval, net_eval, num_classes,
                                                                            image_syn_eval, label_syn_eval, dst_train, dst_test,
                                                                            trainloader, testloader, args, texture=args.texture)
                accs_test.append(acc_test)
                accs_train.append(acc_train)

            accs_test = np.array(accs_test)
            accs_train = np.array(accs_train)
            acc_test_mean = np.mean(accs_test)
            acc_test_std = np.std(accs_test)
            acc_train_mean = np.mean(accs_train)
            acc_train_std = np.std(accs_train)
            mean_acc_all.append(acc_test_mean)
        print('Evaluate %d random %s, train set mean = %.4f std = %.4f' % (
            len(accs_train), model_eval, acc_train_mean, acc_train_std))
        print('Evaluate %d random %s, test set mean = %.4f std = %.4f\n-------------------------' % (
            len(accs_test), model_eval, acc_test_mean, acc_test_std))
    print(mean_acc_all)
    print("Mean test accuracy of 10 ramdom sets:", sum(mean_acc_all)/len(mean_acc_all))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='gzoo2', help='dataset')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--num_eval', type=int, default=10, help='how many networks to evaluate on')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--syn_steps', type=int, default=50, help='how many steps to take on synthetic data')


    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--rotate', action='store_true', help='rotate images for 4 times')
    parser.add_argument('--transpose', action='store_true', help='transpose images for augmentation')
    parser.add_argument('--flip_h', action='store_true', help='flip images horizontally for augmentation')
    parser.add_argument('--flip_v', action='store_true', help='flip images vertically for augmentation')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device visible')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)
