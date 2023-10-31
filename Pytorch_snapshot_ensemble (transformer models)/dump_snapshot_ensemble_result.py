import os
from pdb import set_trace
import numpy as np
import json
import shutil
from os import getcwd
from os.path import join as pjoin
from pprint import pprint
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import pyautogui
pyautogui.FAILSAFE= True

import multiprocessing
from time import sleep

import itertools
import matplotlib.pyplot as plt

font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size'   : 16}
plt.rc('font', **font)

def load_metadata_file(file_path, cls_list):
    print(file_path)
    df = pd.read_csv(file_path, delimiter=',', header=None)
    gt_col = df[0].copy()
    gt_col = [elem.split('/')[0] for elem in gt_col]
    
    # Replace strings
#    gt_col = ["anode" if elem=="AN" else elem for elem in gt_col]
#    gt_col = ["fieldjoint" if elem=="FJ" else elem for elem in gt_col]
#    gt_col = ["freespan" if elem=="SP" else elem for elem in gt_col]
#    gt_col = ["span correction" if elem=="ST" else elem for elem in gt_col]
#    gt_col = ["concrete damage" if elem=="concrete_damage" else elem for elem in gt_col]

#    print(gt_col)    

    gt_col = [cls_list.index(elem) for elem in gt_col]
    gt_col = np.array(gt_col)
    df = df.drop(df.columns[[0]], axis=1)
    mat = df.to_numpy()
    return (mat, gt_col)

def dump_result_using_metadata_file(
        metadata_file_path, cls_list,
         out_confusion_matrix_path, out_result_path):
    (mat, y_true) = load_metadata_file(metadata_file_path, cls_list)
    y_pred = np.argmax(mat, axis=1)
    conf = confusion_matrix(y_true, y_pred)

    np.savetxt(out_confusion_matrix_path, conf, fmt="%d", delimiter=",")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    with open(out_result_path, 'wt') as fid:
        fid.write('Acc,F1\n')
        fid.write('%.5f,%.5f\n' % (acc, f1))
