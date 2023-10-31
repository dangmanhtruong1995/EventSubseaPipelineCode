import os
import pdb
import shutil
import numpy as np
from lxml import etree as ET
from sklearn.model_selection import KFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    dataset_name = 'voc2012'
    base_path = '/truong-evo/datasets/'
    num_of_folds = 5
    if dataset_name == 'voc2012':
        class_list = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep','sofa', 'train', 'tvmonitor']
    else:
        pdb.set_trace()

    num_of_classes = len(class_list)
    class_name_to_idx = {}
    for i in range(len(class_list)):
        class_name_to_idx[class_list[i]] = i
    trainval_path = os.path.join(base_path, dataset_name, 'trainval')

    # Load trainval
    file_list = os.listdir(os.path.join(trainval_path, 'images'))
    file_list = [file_name[:-4] for file_name in file_list]
    num_of_files = len(file_list)
    y = np.zeros((num_of_files, num_of_classes))
    for idx in range(len(file_list)):
        file_name = file_list[idx]

        # Load xml file in order to find all classes within an image
        full_filepath = os.path.join(trainval_path, 'annotations', file_name + '.xml')
        tree = ET.parse(full_filepath)
        root = tree.getroot()
        for member in root.findall('object'):
            bnd_box = member.find('bndbox')
            xmin = int(bnd_box.find('xmin').text)
            ymin = int(bnd_box.find('ymin').text)
            xmax = int(bnd_box.find('xmax').text)
            ymax = int(bnd_box.find('ymax').text)
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member.find('name').text,
                    xmin,
                    ymin,
                    xmax,
                    ymax)
            class_name = member.find('name').text
            # try:
            #     value = (root.find('filename').text,
            #             int(root.find('size')[0].text),
            #             int(root.find('size')[1].text),
            #            member[0].text,
            #            int(member[4][0].text),
            #            int(member[4][1].text),
            #            int(member[4][2].text),
            #            int(member[4][3].text)
            #            )
            #    class_name = member[0].text
            #    # xml_list.append(value)
            #except:
                # VOC datasets contains head parts which we don't need
            #    value = (root.find('filename').text,
            #             int(root.find('size')[0].text),
            #             int(root.find('size')[1].text),
            #            member[0].text,
            #            int(member[-1][0].text),
            #            int(member[-1][1].text),
            #            int(member[-1][2].text),
            #            int(member[-1][3].text)
            #            )
            #    class_name = member[0].text
            ## print('Object: %s' % (class_name))
            class_idx = class_name_to_idx[class_name]
            y[idx, class_idx] = 1
        # pdb.set_trace()
        pass

    # Divide into folds
    kf = KFold(n_splits=num_of_folds)
    fold_idx = 1
    for train_indices, val_indices in kf.split(y):
    # for train_indices, test_indices in MultilabelStratifiedKFold(n_splits=num_of_folds).split(np.zeros((y.shape[0], 1)), y):
        # pass
        y_train = y[train_indices, :]
        y_val = y[val_indices, :]
        print("Train: %s" % (np.sum(y_train, axis=0)))
        print("Val: %s" % (np.sum(y_val, axis=0)))
        print("")

        fold_path = os.path.join(base_path, dataset_name, 'trainval_fold_%d' % (fold_idx))
        fold_path_train = os.path.join(fold_path, 'train')
        fold_path_val = os.path.join(fold_path, 'val')

        if os.path.isdir(fold_path):
            shutil.rmtree(fold_path)
        os.mkdir(fold_path)
        os.mkdir(fold_path_train)
        os.mkdir(fold_path_val)
        os.mkdir(os.path.join(fold_path_train, 'images'))
        os.mkdir(os.path.join(fold_path_train, 'annotations'))
        os.mkdir(os.path.join(fold_path_val, 'images'))
        os.mkdir(os.path.join(fold_path_val, 'annotations'))

        for idx in train_indices:
            file_name = file_list[idx]
            shutil.copyfile(os.path.join(trainval_path, 'images', file_name + '.jpg'),
                os.path.join(fold_path_train, 'images', file_name + '.jpg'))
            shutil.copyfile(os.path.join(trainval_path, 'annotations', file_name + '.xml'),
                os.path.join(fold_path_train, 'annotations', file_name + '.xml'))

        for idx in val_indices:
            file_name = file_list[idx]
            shutil.copyfile(os.path.join(trainval_path, 'images', file_name + '.jpg'),
                os.path.join(fold_path_val, 'images', file_name + '.jpg'))
            shutil.copyfile(os.path.join(trainval_path, 'annotations', file_name + '.xml'),
                os.path.join(fold_path_val, 'annotations', file_name + '.xml'))

        fold_idx += 1
        # pdb.set_trace()

    # for fold_num in range(1, num_of_folds + 1):
    #     fold_path = os.path.join(base_path, dataset_name, 'trainval_fold_%d' % (fold_num))
    #     fold_path_train = os.path.join(fold_path, 'train')
    #     fold_path_val = os.path.join(fold_path, 'val')

