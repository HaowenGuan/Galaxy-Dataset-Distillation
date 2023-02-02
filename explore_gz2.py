import collections

from astropy.io import fits
import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil


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
    classes_l = np.array(classes_l)
    index = np.argmax(classes_l)
    return index, classes_l[index]


if __name__ == '__main__':
    gzoo = fits.open(os.path.join('Galaxy-DR17-dataset/gzoo2', 'zoo2MainSpecz_sizes.fit'))[1].data
    indexes = dict()
    for i, id in enumerate(gzoo['dr7objid']):
        indexes[id] = i
    path = '/data/sbcaesar/image'
    classes_count = collections.defaultdict(int)
    classes_p = collections.defaultdict(list)
    count = 0
    for image in os.listdir(path):
        if ".jpg" not in image:
            continue
        id = int(image[:-4])
        # count += 1
        # if count > 10000: break
        c, p = get_classes(gzoo, indexes, id)
        classes_count[c] += 1
        classes_p[c].append([p, image])

        # if c == 9:
        #     shutil.copyfile(os.path.join(path, image), os.path.join('/data/sbcaesar/class9', image))

    for key in classes_count:
        # cur = '/data/sbcaesar/classes/' + str(key)
        cur = '/data/sbcaesar/classes/1000/'
        classes_p[key].sort()
        # if os.path.exists(cur):
        #     os.path.mkdir(cur)
        for p, image in classes_p[key][-100:]:
            # print(image)
            shutil.copyfile(os.path.join(path, image), os.path.join(cur, image))
        print(key, classes_count[key])

