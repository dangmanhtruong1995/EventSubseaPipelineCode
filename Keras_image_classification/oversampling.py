from pdb import set_trace
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import Callback
from PIL import Image
import os
from os.path import join as pjoin
import numpy as np
import cv2
import shutil
from pprint import pprint
import random

from custom_utils import make_folder as mkdir

def oversample_image_dataset(dataset_name, new_oversampled_dataset_name,
        base_path, subset_name):
    # First, check the number of images in each class
    curr_subset_path = pjoin(base_path, dataset_name, subset_name)
    cls_list = os.listdir(curr_subset_path)
    n_img_per_cls = []
    cls_max_n_img = ""
    max_n_img = -1
    for cls_name in cls_list:
        n_img = len(os.listdir(pjoin(curr_subset_path, cls_name)))
        n_img_per_cls.append(n_img)
        if max_n_img < n_img:
            cls_max_n_img = cls_name
            max_n_img = n_img

    # For the class with highest number of images, simply copy the entire folder
    new_dataset_path = pjoin(base_path,
        new_oversampled_dataset_name
    )
    new_dataset_subset_path = pjoin(new_dataset_path, subset_name)
    mkdir(new_dataset_subset_path)
    shutil.copytree(
        pjoin(curr_subset_path, cls_max_n_img),
        pjoin(new_dataset_subset_path, cls_max_n_img),
    )

    # For the remaining classes, perform oversampling
    for cls_name in cls_list:
        if cls_name == cls_max_n_img:
            continue
        mkdir(pjoin(new_dataset_subset_path, cls_name))
        img_list = os.listdir(pjoin(curr_subset_path, cls_name))
        n_img = len(img_list)
        oversampled_arr = np.zeros(n_img, dtype=np.int32)
        for idx in range(max_n_img):
            # Get random image
            rand_img_idx = random.randint(0, n_img-1)
            rand_img_name = img_list[rand_img_idx]
            oversampled_arr[rand_img_idx] += 1

            rand_img_path = pjoin(curr_subset_path, cls_name, rand_img_name)

            # Copy to oversampled folder
            new_img_name = "%s_oversampled_%d.%s" % (
                rand_img_name[:-4],
                oversampled_arr[rand_img_idx],
                rand_img_name[-3:]
            )
            new_img_path = pjoin(new_dataset_subset_path, cls_name, new_img_name)
            shutil.copyfile(
                rand_img_path,
                new_img_path
            )

    # set_trace()

    pass

def main():
    dataset_name = 'Saibok_2014_2017_train_val_test'
    new_oversampled_dataset_name = 'Saibok_2014_2017_oversampled_train_val_test'
    base_path = '/home/ubuntu/TRUONG/datasets/'
    subset_list = ['train', 'val']

    mkdir(pjoin(base_path, new_oversampled_dataset_name))

    for subset_name in subset_list:
        oversample_image_dataset(dataset_name, new_oversampled_dataset_name,
            base_path, subset_name)

if __name__ == '__main__':
    main()
