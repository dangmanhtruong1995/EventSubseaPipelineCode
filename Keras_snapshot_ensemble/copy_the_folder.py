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

def get_images_as_dict(dataset_path):
    subset_list = ["train", "val", "test"]
    img_dict = {}
    for subset_name in subset_list:
        img_dict[subset_name] = {}
        cls_list = os.listdir(
            pjoin(dataset_path, subset_name))
        for cls_name in cls_list:
            img_dict[subset_name][cls_name] = os.listdir(
                pjoin(dataset_path, subset_name, cls_name))

    return img_dict

def main():
    source_path = "/home/truong/Desktop/TRUONG/datasets/Saibok_BP_balanced_dataset_GammaContrastAdjustment"
    source_path_binary = "/home/truong/Desktop/TRUONG/datasets/Saibok_BP_balanced_binary_dataset"
    dest_path = "/home/truong/Desktop/TRUONG/datasets/Saibok_BP_balanced_dataset"

    binary_cls_list = ["normal", "abnormal"]
    cls_list = ["concrete_damage", "exposure", "debris", "normal"]
    subset_list = ["train", "val", "test"]

    # Get all images to copy in a dict
    img_dict = get_images_as_dict(source_path)
    img_dict_binary = get_images_as_dict(source_path_binary)

    # Copy
    for subset_name in subset_list:
        for cls_name in cls_list:
            for img_name in img_dict[subset_name][cls_name]:
                # If normal, get in normal folder
                # else get in abnormal folder
                if cls_name == "normal":
                    img_path_to_copy_from = pjoin(
                        source_path_binary,
                        subset_name,
                        "normal",
                        img_name,
                    )
                    img_dest_path = pjoin(
                        dest_path,
                        subset_name,
                        "normal",
                        img_name,
                    )
                else:
                    img_path_to_copy_from = pjoin( 
                        source_path_binary,
                        subset_name,
                        "abnormal",
                        img_name,
                    )
                    img_dest_path = pjoin(
                        dest_path,
                        subset_name,
                        cls_name,
                        img_name,
                    )
                copyfile(img_path_to_copy_from, img_dest_path)
#    set_trace()


if __name__ == "__main__":
    main()
