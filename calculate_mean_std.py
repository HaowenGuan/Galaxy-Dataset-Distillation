import torch
import os
import cv2 as cv

path = 'Galaxy-DR17-dataset/MaNGA/gzoo2'
dst_total = []
images_all = []
for image in os.listdir(path):
    if ".jpg" not in image:
        continue
    image_dir = os.path.join(path, image)

    id = int(image[:-4])
    img = cv.imread(image_dir)
    # TODO
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA) / 255
    # TODO
    #img = cv.cvtColor(np.float32(img), cv.COLOR_BGR2GRAY)
    #
    img = torch.from_numpy(img.T)
    images_all.append(torch.unsqueeze(img, dim=0))

images_all = torch.cat(images_all, dim=0)
print(images_all.shape)
for ch in range(3):
    print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))