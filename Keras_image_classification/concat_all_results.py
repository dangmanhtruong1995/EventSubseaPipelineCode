from pdb import set_trace
import os
from os.path import join as pjoin
import numpy as np
import shutil
from pprint import pprint

def concat_all_results(dataset_name, model_list):
    with open("%s_results.csv" % (dataset_name), "wt") as fid:
        model_name = model_list[0]
        file_name = '%s_%s_result.csv' % (dataset_name, model_name)
        with open(file_name) as fid_inp:
            lines = [line.rstrip() for line in fid_inp]
            line = lines[0]
            fid.write("Model,%s\n" % (line))


        for model_name in model_list:
            file_name = '%s_%s_result.csv' % (dataset_name, model_name)
            with open(file_name) as fid_inp:
                lines = [line.rstrip() for line in fid_inp]
                line = lines[1]
                fid.write("%s,%s\n" % (model_name, line))

def main():
    dataset_name = 'FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test'
    model_list = ['VGG16', 'Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
    concat_all_results(dataset_name, model_list)


if __name__ == "__main__":
    main()
