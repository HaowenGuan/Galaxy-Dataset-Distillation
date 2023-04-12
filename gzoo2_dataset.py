import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, ResNet18_AP
from astropy.io import fits
from astropy.table import Table
import cv2 as cv
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.classes = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.uint8)
        target = torch.tensor(self.target[idx], dtype=torch.int)
        return data, target

class GZooDataset:
    def __init__(self, train, test):
        self.train = train
        self.test = test

channel = 3
im_size = (128, 128)
num_classes = 10

mean = [0.0740, 0.0606, 0.0490]
std = [0.1295, 0.1011, 0.0914]

gzoo = fits.open(os.path.join('Galaxy-DR17-dataset/gzoo2', 'zoo2MainSpecz_sizes.fit'))[1].data
indexes = dict()
for i, id in enumerate(gzoo['dr7objid']):
    indexes[id] = i


def get_classes(id):
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


path = '/data/sbcaesar/classes/6000'
dst_train = []
dst_test = []
np.random.seed(1)
for c in range(10):
    class_path = os.path.join(path, str(c))
    class_image_list = os.listdir(class_path)
    np.random.shuffle(class_image_list)
    # Prepare Train Set with rotation Augmentation
    for image in class_image_list[:500]:
        if ".jpg" not in image:
            continue
        image_dir = os.path.join(class_path, image)
        id = int(image[:-4])
        im = Image.open(image_dir)
        aug = 1
        for i in range(aug):
            img = im.rotate((360 // aug) * i)
            img = np.array(img)[:, :, :3]
            img = img[img.shape[0] // 4:(img.shape[0] * 3) // 4, img.shape[1] // 4:(img.shape[1] * 3) // 4]
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
            img = torch.from_numpy(img.T)
            img = transforms.Normalize(mean, std)(img)
            dst_train.append((img, get_classes(id)))
    # Prepare Test Set
    for image in class_image_list[500:]:
        if ".jpg" not in image:
            continue
        image_dir = os.path.join(class_path, image)
        id = int(image[:-4])
        img = cv.imread(image_dir)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Only use when open with cv2
        img = img[img.shape[0] // 4:(img.shape[0] * 3) // 4, img.shape[1] // 4:(img.shape[1] * 3) // 4]
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
        img = torch.from_numpy(img.T)
        img = transforms.Normalize(mean, std)(img)
        dst_test.append((img, get_classes(id)))
print("Generated Augmented Train Set of", len(dst_train), "images.")
print("Processed Test Set of", len(dst_test), "images.")
class_names = [str(i) for i in range(num_classes)]
class_map = {x: x for x in range(num_classes)}