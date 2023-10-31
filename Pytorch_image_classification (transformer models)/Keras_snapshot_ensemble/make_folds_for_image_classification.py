import os
import sys
from pdb import set_trace
# from cv2 import cv2
#import shutil
import numpy as np
#from os import mkdir, listdir
from shutil import copyfile, rmtree
from os.path import join as pjoin
import json
import math
from random import shuffle
#import scipy.io as spio
#from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

#from custom_utils import make_folder as mkdir

def mkdir(folder_path):
    """ Make a new folder. If it already exists, delete it then make a 
        new one

        Parameters:
        ----------
        folder_path: Folder path

        Returns:
        ----------
        None.

    """
    if os.path.exists(folder_path):
        rmtree(folder_path)
    os.mkdir(folder_path)


def get_all_images_for_each_class(root_img_path):
    cls_list = os.listdir(root_img_path)
    all_img_path_dict = {}
    for cls_idx, cls_name in enumerate(cls_list):
        all_img_path_dict[cls_name] = []
        img_path = pjoin(root_img_path, cls_name)
        img_list = os.listdir(img_path)
        for img_name in img_list:
            all_img_path_dict[cls_name].append(img_name)
    return all_img_path_dict

def shuffle_dict(the_dict):
    for key in the_dict:
        shuffle(the_dict[key])

def divide_list_into_folds(the_list, n_fold):
    n_elem = len(the_list)
    folds_in_idx = np.array_split(range(n_elem), n_fold)
    folds = []
    for fold_idx in range(n_fold):
        idx_list = folds_in_idx[fold_idx].tolist()
        folds.append([the_list[idx] for idx in idx_list])
    return folds

def divide_dict_into_folds(the_dict, n_fold, cls_list):
    the_dict_by_folds = {}
    for cls_name in cls_list:
        folds = divide_list_into_folds(the_dict[cls_name], n_fold)
        the_dict_by_folds[cls_name] = folds
    return the_dict_by_folds

def check_folds(folds, n_fold):
    # Test function for divide_list_into_folds
    print("Running check_folds")
    print("n_fold = %d" % (n_fold))
    print("len(folds) = %d" % (len(folds)))
    for fold_idx, fold in enumerate(folds):
        print("folds[%d]:" % (fold_idx))
        print(fold)

def create_cross_validation_folders(dataset_path, n_fold, cls_list):
    for fold_idx in range(n_fold):
        fold_path = pjoin(dataset_path, "train_fold_%d" % (fold_idx+1))
        mkdir(fold_path)
        fold_train_path = pjoin(fold_path, "train")
        mkdir(fold_train_path)
        fold_val_path = pjoin(fold_path, "val")
        mkdir(fold_val_path)
        for cls_name in cls_list:
            fold_train_cls_path = pjoin(fold_train_path, cls_name)
            mkdir(fold_train_cls_path)
            fold_val_cls_path = pjoin(fold_val_path, cls_name)
            mkdir(fold_val_cls_path)

def save_images_to_cross_validation_folders(dataset_path, n_fold, cls_list, all_img_path_dict_by_folds):
    for fold_val_idx in range(n_fold):
        # Write to val folder in the current fold
        for cls_name in cls_list:
            img_list = all_img_path_dict_by_folds[cls_name][fold_val_idx]
            for img_name in img_list:
                source_path = pjoin(dataset_path, "train", cls_name, img_name)
                dest_path = pjoin(dataset_path, "train_fold_%d" % (fold_val_idx+1), "val", cls_name, img_name)
                copyfile(source_path, dest_path)
        del cls_name

        # Write to train folder in the current fold
        for fold_train_idx in range(n_fold):
            if fold_train_idx == fold_val_idx:
                continue
            for cls_name in cls_list:
                img_list = all_img_path_dict_by_folds[cls_name][fold_train_idx]
                for img_name in img_list:
                    source_path = pjoin(dataset_path, "train", cls_name, img_name)
                    dest_path = pjoin(dataset_path, "train_fold_%d" % (fold_val_idx+1), "train", cls_name, img_name)
                    copyfile(source_path, dest_path)
            del cls_name


def main():
    n_fold = 5
#    dataset_name = 'imbalanced_mnist_example_normal_abnormal'
#    dataset_name = 'Saibok_2014_2017_balanced_train_val_test'
#    dataset_name = 'Saibok_Chevron_data_train_val_test'
#    dataset_name = 'MNIST'
    dataset_name = 'Saibok_BP_balanced_dataset'
    base_path = '/home/truong/Desktop/TRUONG/datasets'

    dataset_path = pjoin(base_path, dataset_name)
    train_path = pjoin(dataset_path, "train")
    cls_list = os.listdir(train_path)

    # Get all images for each class in the train set
    all_img_path_dict = get_all_images_for_each_class(train_path)
#     set_trace()

    # Shuffle them
    shuffle_dict(all_img_path_dict)

    # Check if the function works correctly
#    folds = divide_list_into_folds(all_img_path_dict["abnormal"], n_fold)
#    check_folds(folds, n_fold)

    # Divide all the images in each class into n_fold folds
    all_img_path_dict_by_folds = divide_dict_into_folds(all_img_path_dict, n_fold, cls_list)

    # Create cross-validation folders
    create_cross_validation_folders(dataset_path, n_fold, cls_list)

    # Then with each fold, save to train and val sets
    save_images_to_cross_validation_folders(dataset_path, n_fold, cls_list, all_img_path_dict_by_folds)


#    set_trace()



if __name__ == '__main__':
    main()
