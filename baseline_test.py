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


def get_classes(gzoo, indexes, id):

    d = gzoo[indexes[id]]
    class_1 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a16_completely_round_fraction']
    class_2 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a17_in_between_fraction']
    class_3 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a18_cigar_shaped_fraction']
    class_4 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a04_yes_fraction'] * (
                d['t09_bulge_shape_a25_rounded_fraction'] + d['t09_bulge_shape_a26_boxy_fraction'])
    class_5 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a04_yes_fraction'] * d[
        't09_bulge_shape_a27_no_bulge_fraction']
    class_6 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d[
        't03_bar_a06_bar_fraction'] * d['t04_spiral_a08_spiral_fraction']
    class_7 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d[
        't03_bar_a06_bar_fraction'] * d['t04_spiral_a09_no_spiral_fraction']
    class_8 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d[
        't03_bar_a07_no_bar_fraction'] * d['t04_spiral_a08_spiral_fraction']
    class_9 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d[
        't03_bar_a07_no_bar_fraction'] * d['t04_spiral_a09_no_spiral_fraction']
    class_10 = d['t01_smooth_or_features_a03_star_or_artifact_fraction']

    classes_l = [class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10]
    return np.argmax(np.array(classes_l))

def ds_test_on_original():
    # 0.0592 #
    mean = [0.0676, 0.0570, 0.0456]
    # 0.1058 #
    std = [0.1230, 0.0990, 0.0887]

    gzoo = fits.open(os.path.join('Galaxy-DR17-dataset/gzoo2', 'zoo2MainSpecz_sizes.fit'))[1].data
    indexes = dict()
    for i, id in enumerate(gzoo['dr7objid']):
        indexes[id] = i

    # path = '/data/sbcaesar/xuan_galaxy/Galaxy-DR17-dataset/gzoo2/image'
    path = '/data/sbcaesar/classes/1000'

    dst_test = []
    count = 0
    for image in os.listdir(path):
        if ".jpg" not in image:
            continue
        image_dir = os.path.join(path, image)

        count += 1
        if count > 1000: break

        id = int(image[:-4])
        img = cv.imread(image_dir)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img[img.shape[0] // 4: (img.shape[0] * 3) // 4, img.shape[1] // 4: (img.shape[1] * 3) // 4]
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
        # img = cv.cvtColor(np.float32(img), cv.COLOR_BGR2GRAY)
        img = torch.from_numpy(img.T)
        img = transforms.Normalize(mean, std)(img)

        dst_test.append((img, get_classes(gzoo, indexes, id)))
    np.random.shuffle(dst_test)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=2)
    return dst_test, testloader

def get_images_average(c, images_all, indices_class):
     avg_pic = torch.mean(images_all[indices_class[c]],0)
     save_image(avg_pic, 'logs/gzoo2/average_images/class_'+str(c)+"_average.png")
     return avg_pic

def get_random_images(c, images_all, indices_class, n=1):  # get random n images from class c
    idx_shuffle = np.random.permutation(indices_class[c])[:n]
    return images_all[idx_shuffle]

def main(args):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    channel = 3
    im_size = (128, 128)
    num_classes = 10

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]

    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    #
    for ch in range(3):
        print('real images channel %d, mean = %.4f, std = %.4f' % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
    print("Loading test:")

    # dst_test, testloader = ds_test_on_original()
    print("Load test!")
    mean_acc_all = []
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=128, shuffle=False, num_workers=2)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    if args.eval_method == "distilled":
        image_syn = torch.load(args.distilled_path)

    else:
        for c in range(num_classes):
            if args.eval_method == "average":
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images_average(c, images_all, indices_class).detach().data
            elif args.eval_method == "random":
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_random_images(c, images_all, indices_class).detach().data

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    for i in range(10):
        args.lr_net = 0.0001

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

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=10, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.0007626, help='initialization for synthetic learning rate')

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
    parser.add_argument('--syn_steps', type=int, default=50, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--eval_method', type=str, default='distilled', help='evaluation method: distilled, average or random')
    parser.add_argument('--distilled_path', type=str, default='/data/sbcaesar/mac_galaxy/logged_files/CIFAR10/cifar10-1ipc-10-no-mini-duration/images_2600.pt', help='path to your distilled images')

    args = parser.parse_args()

    main(args)
