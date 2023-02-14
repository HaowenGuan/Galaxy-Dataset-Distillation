# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

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
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

config = Config()
np.random.seed(1)

def get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None):
    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'gzoo2_prob':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        #0.0592 #
        mean = [0.0376, 0.0331, 0.0248]
        #0.1058 #
        std = [0.0754, 0.0617, 0.0547]

        gzoo = fits.open(os.path.join('Galaxy-DR17-dataset/gzoo2', 'zoo2MainSpecz_sizes.fit'))[1].data
        indexes = dict()
        for i, id in enumerate(gzoo['dr7objid']):
            indexes[id] = i

        def get_classes(id):
            d = gzoo[indexes[id]]
            class_1 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a16_completely_round_fraction']
            class_2 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a17_in_between_fraction']
            class_3 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a18_cigar_shaped_fraction']
            class_4 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a04_yes_fraction'] * (d['t09_bulge_shape_a25_rounded_fraction'] + d['t09_bulge_shape_a26_boxy_fraction'])
            class_5 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a04_yes_fraction'] * d['t09_bulge_shape_a27_no_bulge_fraction']
            class_6 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a06_bar_fraction'] * d['t04_spiral_a08_spiral_fraction']
            class_7 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a06_bar_fraction'] * d['t04_spiral_a09_no_spiral_fraction']
            class_8 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a07_no_bar_fraction'] * d['t04_spiral_a08_spiral_fraction']
            class_9 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a07_no_bar_fraction'] * d['t04_spiral_a09_no_spiral_fraction']
            class_10 = d['t01_smooth_or_features_a03_star_or_artifact_fraction']

            classes_l = [class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10]
            # return np.array(classes_l)
            return np.argmax(np.array(classes_l))

        path = '/data/sbcaesar/image'
        dst_total = []
        count = 0
        for image in os.listdir(path):
            if ".jpg" not in image:
                continue
            image_dir = os.path.join(path, image)
            count += 1
            if count % 1000 == 0: print(count)
            if count > 10000:
                break

            id = int(image[:-4])
            img = cv.imread(image_dir)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
            img = torch.from_numpy(img.T)
            img = transforms.Normalize(mean, std)(img)

            dst_total.append((img, get_classes(id)))

        np.random.shuffle(dst_total)
        dst_train = dst_total[:int(0.8 * len(dst_total))]
        dst_test = dst_total[int(0.8 * len(dst_total)):]

        class_names = [str(i) for i in range(num_classes)]
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'gzoo2':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        #0.0592 #
        mean = [0.0676, 0.0570, 0.0456]
        #0.1058 #
        std = [0.1230, 0.0990, 0.0887]

        gzoo = fits.open(os.path.join('Galaxy-DR17-dataset/gzoo2', 'zoo2MainSpecz_sizes.fit'))[1].data
        indexes = dict()
        for i, id in enumerate(gzoo['dr7objid']):
            indexes[id] = i

        def get_classes(id):
            d = gzoo[indexes[id]]
            class_1 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a16_completely_round_fraction']
            class_2 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a17_in_between_fraction']
            class_3 = d['t01_smooth_or_features_a01_smooth_fraction'] * d['t07_rounded_a18_cigar_shaped_fraction']
            class_4 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a04_yes_fraction'] * (d['t09_bulge_shape_a25_rounded_fraction'] + d['t09_bulge_shape_a26_boxy_fraction'])
            class_5 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a04_yes_fraction'] * d['t09_bulge_shape_a27_no_bulge_fraction']
            class_6 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a06_bar_fraction'] * d['t04_spiral_a08_spiral_fraction']
            class_7 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a06_bar_fraction'] * d['t04_spiral_a09_no_spiral_fraction']
            class_8 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a07_no_bar_fraction'] * d['t04_spiral_a08_spiral_fraction']
            class_9 = d['t01_smooth_or_features_a02_features_or_disk_fraction'] * d['t02_edgeon_a05_no_fraction'] * d['t03_bar_a07_no_bar_fraction'] * d['t04_spiral_a09_no_spiral_fraction']
            class_10 = d['t01_smooth_or_features_a03_star_or_artifact_fraction']

            classes_l = [class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10]
            return np.argmax(np.array(classes_l))

        path = '/data/sbcaesar/classes'
        dst_train = []
        dst_test = []
        for c in range(10):
            class_total = []
            class_path = os.path.join(path, str(c))
            for image in os.listdir(class_path):
                if ".jpg" not in image:
                    continue
                image_dir = os.path.join(class_path, image)

                id = int(image[:-4])
                img = cv.imread(image_dir)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = img[img.shape[0] // 4: (img.shape[0] * 3) // 4,img.shape[1] // 4: (img.shape[1] * 3) // 4]
                img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
                #img = cv.cvtColor(np.float32(img), cv.COLOR_BGR2GRAY)
                img = torch.from_numpy(img.T)
                img = transforms.Normalize(mean, std)(img)

                class_total.append((img, get_classes(id)))
            np.random.shuffle(class_total)
            dst_train += class_total[:int(0.8 * len(class_total))]
            dst_test += class_total[int(0.8 * len(class_total)):]

        class_names = [str(i) for i in range(num_classes)]
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'dl-DR17':
        channel = 3
        im_size = (128, 128)
        num_classes = 8

        #0.0592 #
        mean = [0.0695364302974106, 0.060510241696901314, 0.04756364403842208]
        #0.1058 #
        std = [0.123113038980545, 0.10351957804657039, 0.09070320107800815]

        dl17 = fits.open(os.path.join('Galaxy-DR17-dataset/MaNGA', 'manga-morphology-dl-DR17.fits'))[1].data
        classes = dict()
        bar_edgeon = []
        for d in dl17:
            if (d['T-Type'] < 0):
                if (d['P_S0'] < 0.5):
                    classes[d['INTID']] = 0
                else:
                    classes[d['INTID']] = 1
            elif (d['T-Type'] > 0 and d['T-Type'] < 3):
                if (d['P_bar'] > 0.8 and d['P_edge'] < 0.8):
                    classes[d['INTID']] = 2
                elif (d['P_bar'] < 0.8 and d['P_edge'] > 0.8):
                    classes[d['INTID']] = 3
                elif (d['P_bar'] > 0.8 and d['P_edge'] > 0.8):
                    bar_edgeon.append(d['INTID'])
                else:
                    classes[d['INTID']] = 4
            else:
                if (d['P_bar'] > 0.8 and d['P_edge'] < 0.8):
                    classes[d['INTID']] = 5
                elif (d['P_bar'] < 0.8 and d['P_edge'] > 0.8):
                    classes[d['INTID']] = 6
                elif (d['P_bar'] > 0.8 and d['P_edge'] > 0.8):
                    bar_edgeon.append(d['INTID'])
                else:
                    classes[d['INTID']] = 7

        bar_edgeon_name = [str(i)+".jpg" for i in bar_edgeon]
        path = 'Galaxy-DR17-dataset/MaNGA/image'
        dst_total = []
        # count = 0
        for image in os.listdir(path):
            if ".jpg" not in image or image in bar_edgeon_name:
                continue
            image_dir = os.path.join(path, image)
            # count += 1
            # if count > 20000:
            #     break

            id = int(image[:-4])
            img = cv.imread(image_dir)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
            # img = cv.cvtColor(np.float32(img), cv.COLOR_BGR2GRAY)
            img = torch.from_numpy(img.T)
            img = transforms.Normalize(mean, std)(img)

            dst_total.append((img, classes[id]))

        np.random.shuffle(dst_total)
        dst_train = dst_total[:int(0.8 * len(dst_total))]
        dst_test = dst_total[int(0.8 * len(dst_total)):]

        class_names = ['E', 'S0', 'S1_bar', 'S1_edge_on', 'S1_none', 'S2_bar', 'S2_edge_on', 'S2_none']
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'dl-DR17-05':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        #0.0592 #
        mean = [0.0695364302974106, 0.060510241696901314, 0.04756364403842208]
        #0.1058 #
        std = [0.123113038980545, 0.10351957804657039, 0.09070320107800815]

        dl17 = fits.open(os.path.join('Galaxy-DR17-dataset/MaNGA', 'manga-morphology-dl-DR17.fits'))[1].data
        classes = dict()
        bar_edgeon = []
        for d in dl17:
            if (d['T-Type'] < 0):
                if (d['P_S0'] < 0.5):
                    classes[d['INTID']] = 0
                else:
                    classes[d['INTID']] = 1
            elif (d['T-Type'] > 0 and d['T-Type'] < 3):
                if (d['P_bar'] > 0.5 and d['P_edge'] < 0.5):
                    classes[d['INTID']] = 2
                elif (d['P_bar'] < 0.5 and d['P_edge'] > 0.5):
                    classes[d['INTID']] = 3
                elif (d['P_bar'] > 0.5 and d['P_edge'] > 0.5):
                    classes[d['INTID']] = 4
                else:
                    classes[d['INTID']] = 5
            else:
                if (d['P_bar'] > 0.5 and d['P_edge'] < 0.5):
                    classes[d['INTID']] = 6
                elif (d['P_bar'] < 0.5 and d['P_edge'] > 0.5):
                    classes[d['INTID']] = 7
                elif (d['P_bar'] > 0.5 and d['P_edge'] > 0.5):
                    classes[d['INTID']] = 8
                else:
                    classes[d['INTID']] = 9

        path = 'Galaxy-DR17-dataset/MaNGA/image'
        dst_total = []
        # count = 0
        for image in os.listdir(path):
            if ".jpg" not in image:
                continue
            image_dir = os.path.join(path, image)
            # count += 1

            id = int(image[:-4])
            img = cv.imread(image_dir)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
            img = torch.from_numpy(img.T)
            img = transforms.Normalize(mean, std)(img)

            dst_total.append((img, classes[id]))

        np.random.shuffle(dst_total)
        dst_train = dst_total[:int(0.8 * len(dst_total))]
        dst_test = dst_total[int(0.8 * len(dst_total)):]

        class_names = ['E', 'S0', 'S1_bar', 'S1_edge_on', 'S1_bar_edge_on', 'S1_none', 'S2_bar', 'S2_edge_on', 'S2_bar_edge_on', 'S2_none']
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'dl-DR17-ttype':
        channel = 3
        im_size = (69, 69)
        num_classes = 13

        #0.0592 #
        mean = [0.0695364302974106, 0.060510241696901314, 0.04756364403842208]
        #0.1058 #
        std = [0.123113038980545, 0.10351957804657039, 0.09070320107800815]

        dl17 = fits.open(os.path.join('Galaxy-DR17-dataset/MaNGA', 'manga-morphology-dl-DR17.fits'))[1].data
        classes = dict()
        for d in dl17:
            classes[d['INTID']] = d['T-Type']

        path = 'Galaxy-DR17-dataset/MaNGA/image'
        dst_total = []
        # count = 0
        for image in os.listdir(path):
            image_dir = os.path.join(path, image)
            if os.path.isdir(image_dir):
                continue
            # count += 1

            id = int(image[:-4])
            img = cv.imread(image_dir)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (69, 69), interpolation=cv.INTER_AREA) / 255
            img = torch.from_numpy(img.T)
            img = transforms.Normalize(mean, std)(img)
            dst_total.append((img, classes[id]))

        np.random.shuffle(dst_total)
        dst_train = dst_total[:int(0.8 * len(dst_total))]
        dst_test = dst_total[int(0.8 * len(dst_total)):]

        class_names = ['E', 'S0', 'S1_bar', 'S1_edge_on', 'S1_none', 'S2_bar', 'S2_edge_on', 'S2_none']
        class_map = {x:x for x in range(num_classes)}
    
    elif dataset == 'dl-DR17-Pipe3D':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        #0.0592 #
        mean = [0.0695364302974106, 0.060510241696901314, 0.04756364403842208]
        #0.1058 #
        std = [0.123113038980545, 0.10351957804657039, 0.09070320107800815]

        dr17_DL = fits.open(os.path.join('Galaxy-DR17-dataset/MaNGA', 'manga-morphology-dl-DR17.fits'))[1].data
        df_DL = Table(dr17_DL).to_pandas()
        df_DL = df_DL[['INTID', 'MANGA_ID']]
        df_DL['MANGA_ID'] = df_DL['MANGA_ID'].apply(lambda x : x.strip())
        df_DL = df_DL.drop_duplicates(subset=['MANGA_ID'])

        dr17_Pipe3D = fits.open(os.path.join('Galaxy-DR17-dataset/MaNGA', 'SDSS17Pipe3D_v3_1_1.fits'))[1].data
        df_Pipe3D = Table(dr17_Pipe3D).to_pandas()
        df_Pipe3D['MANGA_ID'] = df_Pipe3D['mangaid']
        df_Pipe3D['log_vel_sigma_Re'] = df_Pipe3D['vel_sigma_Re'].apply(lambda x : np.log(x))
        df_Pipe3D['log_vel_disp_ssp_cen'] = df_Pipe3D['vel_disp_ssp_cen'].apply(lambda x : np.log(x))
        df_Pipe3D = df_Pipe3D[['MANGA_ID', 'log_vel_sigma_Re', 'log_SFR_ssp', 'log_vel_disp_ssp_cen']]
        df_Pipe3D = df_Pipe3D.drop_duplicates(subset=['MANGA_ID'])

        df = pd.merge(df_DL, df_Pipe3D, how = 'inner', on = 'MANGA_ID')

        classes = dict()
        target = 'log_vel_sigma_Re'
        # target = 'log_SFR_ssp'
        # target = 'log_vel_disp_ssp_cen'
        percentiles = np.nanpercentile(list(df[target]), np.linspace(0, 100, num_classes + 1))
        for _, d in df.iterrows():
            for i in range(num_classes):
                if d[target] <= percentiles[i + 1]:
                    classes[d['INTID']] = i
                    break

        path = 'Galaxy-DR17-dataset/MaNGA/image'
        dst_total = []
        for image in os.listdir(path):
            if ".jpg" not in image or int(image[:-4]) not in classes:
                continue
            image_dir = os.path.join(path, image)

            id = int(image[:-4])
            img = cv.imread(image_dir)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
            # img = cv.cvtColor(np.float32(img), cv.COLOR_BGR2GRAY)
            img = torch.from_numpy(img.T)
            img = transforms.Normalize(mean, std)(img)

            dst_total.append((img, classes[id]))

        np.random.shuffle(dst_total)
        dst_train = dst_total[:int(0.8 * len(dst_total))]
        dst_test = dst_total[int(0.8 * len(dst_total)):]

        class_names = [str(i) for i in range(num_classes)]
        class_map = {x:x for x in range(num_classes)}

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val", "images"), transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'ImageNet':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        config.img_net_classes = config.dict[subset]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size)])

        dst_train = datasets.ImageNet(data_path, split="train", transform=transform) # no augmentation
        dst_train_dict = {c : torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.equal(dst_train.targets, config.img_net_classes[c])))) for c in range(len(config.img_net_classes))}
        dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
        loader_train_dict = {c : torch.utils.data.DataLoader(dst_train_dict[c], batch_size=batch_size, shuffle=True, num_workers=16) for c in range(len(config.img_net_classes))}
        dst_test = datasets.ImageNet(data_path, split="val", transform=transform)
        dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
        for c in range(len(config.img_net_classes)):
            dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
            dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c
        print(dst_test.dataset)
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}
        class_names = None


    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}

    else:
        exit('unknown dataset: %s'%dataset)

    if args.zca:
        images = []
        labels = []
        print("Train ZCA")
        for i in tqdm.tqdm(range(len(dst_train))):
            im, lab = dst_train[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        zca.fit(images)
        zca_images = zca(images).to("cpu")
        dst_train = TensorDataset(zca_images, labels)

        images = []
        labels = []
        print("Test ZCA")
        for i in tqdm.tqdm(range(len(dst_test))):
            im, lab = dst_test[i]
            images.append(im)
            labels.append(lab)
        images = torch.stack(images, dim=0).to(args.device)
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        zca_images = zca(images).to("cpu")
        dst_test = TensorDataset(zca_images, labels)

        args.zca_trans = zca


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=128, shuffle=False, num_workers=2)


    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), dist=True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'galaxy':
        net = nn.Sequential(
            nn.Conv2d(3, 32, (6, 6), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, (5, 5), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, (2, 2), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(128, 128, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.25),

            ## Dense layers
            nn.Flatten(1, -1),

            nn.Linear(36992, 128),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.Dropout(0.5),

            nn.Linear(64, 13)
            # nn.Softmax(dim=1)
        )
    elif model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)


    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')


    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda:0'
            # if gpu_num>1:
            #     net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)

        if mode == "train" and texture:
            img = torch.cat([torch.stack([torch.roll(im, (torch.randint(args.im_size[0]*args.canvas_size, (1,)), torch.randint(args.im_size[0]*args.canvas_size, (1,))), (1,2))[:,:args.im_size[0],:args.im_size[1]] for im in img]) for _ in range(args.canvas_samples)])
            lab = torch.cat([lab for _ in range(args.canvas_samples)])

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        if args.dataset == "ImageNet" and mode != "train":
            lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)

        n_b = lab.shape[0]

        output = net(img)

        loss = criterion(output, lab)

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def epoch_regression(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, num_exp = 0, 0
    net = net.to(args.device)
    softmax = nn.Softmax(dim=1)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].float().to(args.device)

        if mode == "train" and texture:
            img = torch.cat([torch.stack([torch.roll(im, (torch.randint(args.im_size[0]*args.canvas_size, (1,)), torch.randint(args.im_size[0]*args.canvas_size, (1,))), (1,2))[:,:args.im_size[0],:args.im_size[1]] for im in img]) for _ in range(args.canvas_samples)])
            lab = torch.cat([lab for _ in range(args.canvas_samples)])

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        n_b = lab.shape[0]

        output = net(img)
        output = softmax(output)

        loss = criterion(output, lab)

        loss_avg += loss.item()*n_b
        num_exp += n_b
        # print(output[0], lab[0])
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp

    return loss_avg


def evaluate_synset(it, it_eval, net, num_classes, images_train, labels_train, dst_test, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in tqdm.tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))
    
    train_cf, test_cf = np.array([[0] * num_classes for _ in range(num_classes)]), np.array([[0] * num_classes for _ in range(num_classes)])
    for dataset, name, count, cf_matrix in [[trainloader, "train", len(dst_train), train_cf],[testloader, "test", len(dst_test), test_cf]]:
        # total_acc = torch.zeros(num_classes)

        pred = []
        true = []
        for i_batch, datum in enumerate(dataset):
            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)

            output = net(img)
            output = torch.argmax(output, 1)

            pred += output.tolist()
            true += lab.tolist()

            # for i in range(lab.shape[0]):
            #     if output[i] == lab[i]:
            #         total_acc[lab[i]] += 1
        
        # print(name, "set ACC of each class", total_acc / count)

        cf_matrix += confusion_matrix(true, pred)
        # print(cf_matrix)
        # df_cm = pd.DataFrame(cf_matrix, index = [i for i in class_names], 
        #     columns = [i for i in class_names])
        # plt.figure(figsize = (12,7))
        # sn.heatmap(df_cm, annot=True, fmt='g')
        # plt.title('Confusion Matrix Iteration{} Evaluation{} {}'.format(it, it_eval, name))
        # plt.xlabel("Prediction")
        # plt.ylabel("True Label")
        # plt.savefig('./cf_matrix_distill/cf_iteration{}_evaluation{}_{}.png'.format(it, it_eval, name))
    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test, train_cf, test_cf
    else:
        return net, acc_train_list, acc_test, train_cf, test_cf


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        model_eval_pool = ['ConvNet', 'AlexNet', 'VGG11', 'ResNet18_AP', 'ResNet18']
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}