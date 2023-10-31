from pdb import set_trace
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import os
from os.path import join as pjoin
import numpy as np
import cv2
import shutil
from pprint import pprint

from custom_utils import make_folder as mkdir

def show_misclassified(base_file_path, model_name, pred_file_name,
        output_folder):
    mkdir(output_folder)
    with open(pred_file_name) as fid:
        lines = [line.rstrip() for line in fid]
    for line in lines:
        splitted = line.split(",")
        img_name = splitted[0]
        gt_cls = splitted[1]
        pred_cls = splitted[2]
        if gt_cls != pred_cls:
            img_path = pjoin(base_file_path, img_name)
            print(img_path)
#            set_trace()
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.title("Ground truth: %s. Predicted: %s" % (gt_cls, pred_cls))
            output_full_path = pjoin(output_folder, "%s_misclassified.png" % (img_name[:-4].split('/')[1]))
#            set_trace()
            plt.savefig(output_full_path)
            plt.close()

def main():
    dataset_path = '/home/ubuntu/TRUONG/datasets/'
#    dataset_list = ['Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test']
    dataset_list = ['Saibok_2014_2017_oversampled_train_val_test']
#    dataset_list = ['KerasOCR+Inpaint+Threshold+SCB+GammaAdjustment+AutomaticBrightnessAndContrast_Saibok_2014_2017_balanced_train_val_test']
#    model_list = ['VGG16', 'Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
    model_list = ['VGG16']
    for dataset_name in dataset_list:
        for model_name in model_list:
            pred_file_name = '%s_%s_predictions.txt' % (dataset_name, model_name)
            output_folder = './%s_%s_misclassified' % (dataset_name, model_name)
            print("MODEL NAME: %s, dataset name: %s" % (model_name, dataset_name))
            show_misclassified(dataset_path, dataset_name, model_name,
                pred_file_name, output_folder)

if __name__ == '__main__':
    main()
