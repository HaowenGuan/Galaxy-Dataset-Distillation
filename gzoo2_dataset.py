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
num_classes = 9

mean = [0.0635, 0.0561, 0.0446]
std = [0.1120, 0.0948, 0.0833]

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
    for c in range(9):
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
            dst_test["labels"].append(c)

        # Prepare Train Set with rotation Augmentation
        for image in class_image_list[100:]:
            if ".jpg" not in image:
                continue
            image_dir = os.path.join(class_path, image)
            id = int(image[:-4])
            im = Image.open(image_dir)
            for i in range(aug):
                img = im.rotate((360 // aug) * i)
                img = np.array(img)[:, :, :3]
                img = img[img.shape[0] // 4:(img.shape[0] * 3) // 4, img.shape[1] // 4:(img.shape[1] * 3) // 4]
                img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
                img = torch.from_numpy(img.T)
                img = transforms.Normalize(mean, std)(img)
                dst_train["images"].append(img)
                dst_train["labels"].append(c)

    train_dataset = CustomDataset(dst_train["images"], dst_train["labels"])
    test_dataset = CustomDataset(dst_test["images"], dst_test["labels"])
    gzoo_dataset = GZooDataset(train_dataset, test_dataset)
    if aug == 1:
        torch.save(gzoo_dataset, os.path.join(path, "gzoo_dataset.pt"))
    else:
        torch.save(gzoo_dataset, os.path.join(path, "gzoo_dataset_aug.pt"))

    print("Generated Augmented Train Set of", len(dst_train["images"]), "images.")
    print("Processed Test Set of", len(dst_test["images"]), "images.")
    class_names = [str(i) for i in range(num_classes)]
    class_map = {x: x for x in range(num_classes)}

if __name__ == '__main__':
    path = "/data/sbcaesar/gzoo2_500ipc"
    build(path)
    # gzoo_dataset = torch.load(os.path.join(path, "gzoo_dataset.pt"))
    # print(gzoo_dataset.train[0])
