import os
from os.path import join as pjoin
import pdb
from time import time
import numpy as np
import copy
import pandas
import scipy
from pdb import set_trace
# import xlsxwriter
# import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import pyautogui
pyautogui.FAILSAFE= True

import multiprocessing
from time import sleep

import itertools

font = {
    # 'family' : 'normal',
    # 'weight' : 'bold',
    'size'   : 16}
plt.rc('font', **font)

def get_dataset_information(dataset_name):
    info = {}
    if dataset_name == 'Saibok_Total_balanced':
        cls_list = ["anode", "debris", "exposure", "freespan", "normal"]
    elif dataset_name == 'Saibok_Chevron_dataset':
        cls_list = ["anode", "fieldjoint", "freespan", "span correction", "normal"]
    elif dataset_name == 'Saibok_BP_balanced_dataset':
        cls_list = ["concrete damage", "debris", "exposure", "normal"]
    info["cls_list"] = cls_list
    return info

def plt_set_fullscreen():
    backend = str(plt.get_backend())
    mgr = plt.get_current_fig_manager()
    if backend == 'TkAgg':
        if os.name == 'nt':
            mgr.window.state('zoomed')
        else:
            mgr.resize(*mgr.window.maxsize())
    elif backend == 'wxAgg':
        mgr.frame.Maximize(True)
    elif (backend == 'Qt4Agg') or (backend == 'QtAgg'):
        mgr.window.showMaximized()

def draw_graph_then_save_and_close_automatically(target=None, args=[]):
    # coords_close_graph = (1365, 12) # Auto click to close graph
    # coords_close_graph = (1895, 11)
    # coords_close_graph = (3791, 96)
#    coords_close_graph = (1884, 15)
#    coords_close_graph = (1906, 41)
    coords_close_graph = (1900, 50)
    multiprocessing.Process(target=target, args=args).start()
    sleep(5)
    pyautogui.moveTo(coords_close_graph)
    pyautogui.click()

def plot_confusion_matrix(cm,classes, output_path, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    plt.figure(figsize=(15,10))
    fig = plt.gcf()
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    # plt.title(title)
    plt.colorbar()

    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0, fontsize=16)
    plt.yticks(tick_marks,classes,rotation=0, fontsize=16)

    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        cm=np.around(cm,decimals=2)
        cm[np.isnan(cm)]=0.0
        print('Normalized confusion matrix')

    else:
        print('Confusion matrix, without normalization')

    thresh=cm.max()/2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",fontsize=26,

                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

        plt.ylabel('True label',fontsize=16)

        plt.xlabel('Predicted label',fontsize=16)

    plt.tight_layout(pad=0)
    plt_set_fullscreen()
    plt.show()
    fig.savefig(output_path, 
        # format="svg",
        bbox_inches="tight")
    plt.close()

def draw_confusion_matrix_func(conf, cls_list, output_path):
    # fig = plt.gcf()
    fig, ax = plt.subplots(figsize=(10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
        display_labels=cls_list)
    disp.plot(ax=ax)    
    # plt.tight_layout(pad=0)
    plt_set_fullscreen()
    plt.show()
    
    fig.savefig(output_path, format="svg",)
    plt.close()

def main():
#    base_path = r"C:\TRUONG\PhD_projects\RSE_pipeline_cnn_paper_2023\Ket_qua\Ket_qua_snapshot_ensemble_RSE_paper_29_08_2023"
    base_path = "/home/truong/Desktop/TRUONG/code/Pytorch_snapshot_ensemble"
#    dataset_list = ['Saibok_Total_balanced', 'Saibok_Chevron_dataset', 'Saibok_BP_balanced_dataset']
    dataset_list = ['Saibok_Total_balanced']
    # dataset_list = ['Saibok_Chevron_dataset', 'Saibok_BP_balanced_dataset']
    clfs_list = [
#        "VGG16",
#        "Resnet50",
#        "InceptionV3",
#        "InceptionResNetV2",
#        "DenseNet121",
#        "XCeption",
        "VisionTransformer",
#        "MaxViT",
#        "SwinTransformer",        
    ]
    subset_list_1 = [
        # "Fold_1",
        # "Fold_2",
        # "Fold_3",
        # "Fold_4",
        # "Fold_5",
        # "val",
        "Test",        
    ]    
    subset_list_2 = [
        # "train_fold_1",
        # "train_fold_2",
        # "train_fold_3",
        # "train_fold_4",
        # "train_fold_5",
        # "val",
        "test",        
    ]
#    snapshot_list = range(40, 440, 40)
    snapshot_list = range(2, 22, 2)
    for dataset_name in dataset_list:
        dataset_path = pjoin(base_path, "SNAPSHOT_RESULT_%s" % (dataset_name),)
        for clfs_name in clfs_list:
            for snapshot_num in snapshot_list:
                for subset_name_1, subset_name_2 in zip(subset_list_1, subset_list_2):
                    info = get_dataset_information(dataset_name)
                    cls_list = info["cls_list"]
                    n_cls = len(cls_list)
                    confmat_path = pjoin(dataset_path,
                            clfs_name,
                            "Result_%s_snapshot_sumrule_epoch_%d" % (clfs_name, snapshot_num),
                            "%s_%s_conf.csv" % (dataset_name, clfs_name),
                    )
                    output_path = pjoin(
                        dataset_path,
                        clfs_name,
                        "Result_%s_snapshot_sumrule_epoch_%d" % (clfs_name, snapshot_num),
                        # "%s_%s_conf_visualize.svg" % (dataset_name, clfs_name), 
                        "%s_%s_conf_visualize.png" % (dataset_name, clfs_name), 
                    )
                    conf = np.loadtxt(confmat_path, delimiter=",")
                    conf = conf.astype(np.int32)
                    draw_graph_then_save_and_close_automatically(
                        # target=draw_confusion_matrix_func,
                        target=plot_confusion_matrix,
                        args=[conf, cls_list, output_path],
                    )
      
      
if __name__ == '__main__':
    main()
