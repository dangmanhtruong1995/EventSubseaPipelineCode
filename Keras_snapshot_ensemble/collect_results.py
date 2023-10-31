import json
import os
from os.path import join as pjoin
import time
import glob
import pandas as pd
import numpy as np
from pdb import set_trace
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def get_acc_f1(file_path):
    with open(file_path) as fid:
        lines = [line.rstrip() for line in fid]
        line = lines[1]
        splitted = line.split(",")
        acc = float(splitted[0])
        f1 = float(splitted[1])
    return (acc, f1)

def main():
    dataset_path = '/home/ubuntu/TRUONG/datasets/'
#    dataset_name = 'FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test'
#    dataset_name = 'Saibok_2014_2017_balanced_train_val_test'
    dataset_name = 'Saibok_GammaAdjustment'
#    output_path_root = './SNAPSHOT_RESULT_ORIG'
    output_path_root = './SNAPSHOT_RESULT_GAMMA'
    model_list = [
#        'VGG11',
#        'VGG16',
#        'ShuffleNetV2',
#        'DenseNet121',
#        'MobileNetV2',
#        'Resnet50',
#        'InceptionV3',
#        'InceptionResNetV2',
#        'DenseNet121',
        'XCeption',
    ]
    n_snapshot = 30
    n_epoch_per_snapshot = 40
    n_base_snapshots_list = np.arange(1, n_snapshot+1).tolist()
    for model_name in model_list:
        output_path_base = pjoin(output_path_root, "%s" % (model_name))

        arr = np.zeros((n_snapshot, 2))
        arr_single = np.zeros((n_snapshot, 2))
        idx = 0

        # Get single and sumrule cumulative results (Accuracy and F1-score)
        for n_base_snapshots in n_base_snapshots_list:
            print("Model name: %s, number of snapshots: %d" % (model_name, n_base_snapshots))

            # Get single snapshot results
            curr_path = pjoin(
                output_path_base,
                "Result_%s_snapshot_epoch_%d" % (model_name, n_base_snapshots*n_epoch_per_snapshot),
                "%s_%s_result.csv" % (dataset_name, model_name),
            )
            (acc_single, f1_single) = get_acc_f1(curr_path)

            # Get cumulative results
            curr_path = pjoin(
                output_path_base,
                "Result_%s_snapshot_sumrule_epoch_%d" % (model_name, n_base_snapshots*n_epoch_per_snapshot),
                "%s_%s_result.csv" % (dataset_name, model_name),
            )
            (acc, f1) = get_acc_f1(curr_path)

            arr[idx, 0] = acc
            arr[idx, 1] = f1
            arr_single[idx, 0] = acc_single
            arr_single[idx, 1] = f1_single
            idx += 1

        df = pd.DataFrame(data = arr,
                        index = n_base_snapshots_list,
                        columns = ["Accuracy", "F1 score"])
        df.to_excel("Result_cumulative_%s.xlsx" % (model_name), index=True, header=True)
        df = pd.DataFrame(data = arr_single,
                        index = n_base_snapshots_list,
                        columns = ["Accuracy", "F1 score"])
        df.to_excel("Result_single_%s.xlsx" % (model_name), index=True, header=True)

        # Compare accuracy and F1 score between single and sumrule cumulative
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(n_base_snapshots_list, arr_single[:, 0], label='Accuracy', color='b')
        ax.plot(n_base_snapshots_list, arr[:, 0], label='Accuracy (cumulative)', color='r')
        plt.legend()
        ax.set_xlabel("Snapshot")
        ax.set_ylabel("Accuracy")
        plt.title('Accuracy')
        plt.show()
        fig.savefig("Result_accuracy_%s.png" % (model_name))
        plt.close()

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(n_base_snapshots_list, arr_single[:, 1], label='F1 score', color='b')
        ax.plot(n_base_snapshots_list, arr[:, 1], label='F1 score (cumulative)', color='r')
        plt.legend()
        ax.set_xlabel("Snapshot")
        ax.set_ylabel("F1 score")
        plt.title('F1 score')
        plt.show()
        fig.savefig("Result_f1_%s.png" % (model_name))
        plt.close()


if __name__ == '__main__':
    main()
