import numpy as np
import os
from torchvision import datasets, transforms
from astropy.io import fits
import cv2 as cv
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.classes = None

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

class GZooDataset(CustomDataset):
    def __init__(self, train, test):
        self.train = train
        self.test = test

channel = 3
im_size = (128, 128)
num_classes = 10

mean = [0.0735, 0.0600, 0.0482]
std = [0.1279, 0.0992, 0.0892]

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


def build(dataset_path: str, aug=1):
    """
    Build the dataset
    :param dataset_path: The dataset path
    :param aug: The number of augmentation for image
    :return: None
    Save a '.pt' file in the dataset folder
    """
    path = dataset_path
    dst_train = {"images": [], "labels": []}
    dst_test = {"images": [], "labels": []}
    np.random.seed(1)
    for c in range(10):
        class_path = os.path.join(path, str(c))
        class_image_list = os.listdir(class_path)
        np.random.shuffle(class_image_list)
        # Prepare Test Set
        for image in class_image_list[:100]:
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
            dst_test["images"].append(img)
            dst_test["labels"].append(get_classes(id))

        # Prepare Train Set with rotation Augmentation
        if c != 9:
            for image in class_image_list[100:]:
                if ".jpg" not in image:
                    continue
                image_dir = os.path.join(class_path, image)
                id = int(image[:-4])
                img_class = get_classes(id)
                im = Image.open(image_dir)
                for i in range(aug):
                    img = im.rotate((360 // aug) * i)
                    img = np.array(img)[:, :, :3]
                    img = img[img.shape[0] // 4:(img.shape[0] * 3) // 4, img.shape[1] // 4:(img.shape[1] * 3) // 4]
                    img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
                    img = torch.from_numpy(img.T)
                    img = transforms.Normalize(mean, std)(img)
                    dst_train["images"].append(img)
                    dst_train["labels"].append(img_class)
        else:
            for image in class_image_list[100:]:
                if ".jpg" not in image:
                    continue
                image_dir = os.path.join(class_path, image)
                id = int(image[:-4])
                img_class = get_classes(id)
                im = Image.open(image_dir)
                im_trans = im.transpose(Image.TRANSPOSE)
                for i in range(aug):
                    img = im.rotate((360 // aug) * i)
                    img = np.array(img)[:, :, :3]
                    img = img[img.shape[0] // 4:(img.shape[0] * 3) // 4, img.shape[1] // 4:(img.shape[1] * 3) // 4]
                    img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
                    img = torch.from_numpy(img.T)
                    img = transforms.Normalize(mean, std)(img)
                    dst_train["images"].append(img)
                    dst_train["labels"].append(img_class)

                    img_trans = im_trans.rotate((360 // aug) * i)
                    img_trans = np.array(img_trans)[:, :, :3]
                    img_trans = img_trans[img_trans.shape[0] // 4:(img_trans.shape[0] * 3) // 4, img_trans.shape[1] // 4:(img_trans.shape[1] * 3) // 4]
                    img_trans = cv.resize(img_trans, (128, 128), interpolation=cv.INTER_AREA) / 255
                    img_trans = torch.from_numpy(img_trans.T)
                    img_trans = transforms.Normalize(mean, std)(img_trans)
                    dst_train["images"].append(img_trans)
                    dst_train["labels"].append(img_class)

    train_dataset = CustomDataset(dst_train["images"], dst_train["labels"])
    test_dataset = CustomDataset(dst_test["images"], dst_test["labels"])
    gzoo_dataset = GZooDataset(train_dataset, test_dataset)
    torch.save(gzoo_dataset, os.path.join(path, "gzoo_dataset.pt"))

    print("Generated Augmented Train Set of", len(dst_train["images"]), "images.")
    print("Processed Test Set of", len(dst_test["images"]), "images.")
    class_names = [str(i) for i in range(num_classes)]
    class_map = {x: x for x in range(num_classes)}

if __name__ == '__main__':
    path = "/data/sbcaesar/classes/6000"
    build(path)
    # gzoo_dataset = torch.load(os.path.join(path, "gzoo_dataset.pt"))
    # print(gzoo_dataset.train[0])
