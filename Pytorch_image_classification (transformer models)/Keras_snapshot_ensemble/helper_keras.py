from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.python.keras.metrics import Metric
#from keras import backend as K
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from pdb import set_trace
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from PIL import Image
import os
from os.path import join as pjoin
import numpy as np
import cv2
import shutil
from pprint import pprint

from custom_utils import make_folder as mkdir

def freeze_all_layers(model):
    """Freeze model weights in every layer."""
    for layer in model.layers:
        layer.trainable = False

        if isinstance(layer, tf.keras.models.Model):
            freeze(layer)

def get_cnn_model(model_name, img_shape):
    if model_name == 'VGG16':
        base_model = tf.keras.applications.VGG16(input_shape=img_shape, \
                                                   include_top=False, \
                                                   weights='imagenet')
    elif model_name == 'VGG19':
        base_model = tf.keras.applications.VGG19(input_shape=img_shape, \
                                                   include_top=False, \
                                                   weights='imagenet')
    elif model_name == 'Resnet50':
        base_model = tf.keras.applications.ResNet50(input_shape=img_shape, 
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'Resnet101':
        base_model = tf.keras.applications.ResNet101(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'Resnet152':
        base_model = tf.keras.applications.ResNet152(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'InceptionV3':
        base_model = tf.keras.applications.InceptionV3(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'InceptionResNetV2':
        base_model = tf.keras.applications.InceptionResNetV2(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'DenseNet121':
        base_model = tf.keras.applications.DenseNet121(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'DenseNet169':
        base_model = tf.keras.applications.DenseNet169(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'DenseNet201':
        base_model = tf.keras.applications.DenseNet201(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    elif model_name == 'XCeption':
        base_model = tf.keras.applications.Xception(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    else:
        raise Exception('In run_cnn: Model name not supported!')
    return base_model

def get_preprocess_input(model_name):
    if model_name == 'VGG16':
        return vgg16_preprocess_input
    elif model_name == 'Resnet50':
        return resnet_preprocess_input
    elif model_name == 'Resnet101':
        return resnet_preprocess_input
    elif model_name == 'InceptionV3':
        return inceptionv3_preprocess_input
    elif model_name == 'InceptionResNetV2':
        return inceptionresnetv2_preprocess_input
    elif model_name == 'DenseNet121':
        return densenet121_preprocess_input
    elif model_name == 'XCeption':
        return xception_preprocess_input
    else:
        raise Exception('In get_preprocess_input: Model name not supported!')    

def get_image_size(model_name):
    if model_name == 'VGG16':
        image_size = 160
    elif model_name == 'VGG19':
        image_size = 299
    elif model_name == 'Resnet50':
        image_size = 224
    elif model_name == 'Resnet101':
        image_size = 224
    elif model_name == 'Resnet152':
        image_size = 224
    elif model_name == 'InceptionV3':
        image_size = 299
    elif model_name == 'InceptionResNetV2':
        image_size = 299
    elif model_name == 'DenseNet121':
        image_size = 224
    elif model_name == 'DenseNet169':
        image_size = 224
    elif model_name == 'DenseNet201':
        image_size = 224
    elif model_name == 'XCeption':
        image_size = 299
    else:
        raise Exception('In run_cnn: Model name not supported!')
    return image_size

def make_datagen(preprocess_input):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
                                   preprocessing_function=preprocess_input,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=True,
                                   vertical_flip=True,
#                                   fill_mode='reflect',
#                                   data_format='channels_last',
#                                   brightness_range=[0.5, 1.5]
    )
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    return (train_datagen, val_datagen, test_datagen)

def make_generators(train_dir, val_dir, test_dir, preprocess_input,
         image_size, batch_size):
    """Function to make train, validation and test generators """
    (train_datagen, val_datagen, test_datagen) = make_datagen(preprocess_input)

    train_generator = train_datagen.flow_from_directory(
                    train_dir,  # Source directory for the training images
#                    target_size=(image_size, image_size),
                    shuffle=True,
                    batch_size=batch_size,
                    seed=0,
                    )
    val_generator = val_datagen.flow_from_directory(
                    val_dir,  # Source directory for the validation images
#                    target_size=(image_size, image_size),
                    shuffle=False,
                    batch_size=batch_size,
                    seed=0,
                    )
    test_generator = test_datagen.flow_from_directory(
                    test_dir, # Source directory for the test images
#                    target_size=(image_size, image_size),
                    shuffle=False,
                    batch_size=1,
                    seed=0,
                    )

#    label_list = train_generator.class_indices
#    numeric_to_class = {}
#    for key, val in label_list.items():
#        numeric_to_class[val] = key
    n_cls = len(train_generator.class_indices.keys())
    cls_list = []
    for cls_idx in range(n_cls):
        for cls_name in train_generator.class_indices.keys():
            if train_generator.class_indices[cls_name] == cls_idx:
                cls_list.append(cls_name)

    return (train_generator, val_generator, test_generator, cls_list)


def save_result(dataset_name, model_name, y_pred, test_generator, output_path):
    y_true = test_generator.classes
#    print('Confusion Matrix')
#    print(confusion_matrix(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    conf = confusion_matrix(y_true, y_pred)

    n_cls = len(test_generator.class_indices.keys())
    cls_list = []
    for cls_idx in range(n_cls):
        for cls_name in test_generator.class_indices.keys():
            if test_generator.class_indices[cls_name] == cls_idx:
                cls_list.append(cls_name)

    np.savetxt(pjoin(output_path, '%s_%s_conf.csv' % (dataset_name, model_name)),
         conf, fmt='%d', delimiter=',')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                               display_labels=cls_list)
    disp.plot()
    plt.show()
    plt.savefig(pjoin(output_path, '%s_%s_conf_visualize.png' % (dataset_name, model_name)))
    plt.close()

    with open(pjoin(output_path, '%s_%s_result.csv' % (dataset_name, model_name)), 'wt') as fid:
        fid.write('Acc,F1\n')
        fid.write('%.5f,%.5f\n' % (acc, f1))

    test_file_list = test_generator.filenames
    with open(pjoin(output_path, '%s_%s_predictions.txt' % (dataset_name, model_name)), 'wt') as fid:
        for (file_name, y_true_elem, y_pred_elem) in zip(test_file_list, y_true, y_pred):
            y_true_elem_cls = cls_list[y_true_elem]
            y_pred_elem_cls = cls_list[y_pred_elem]
            fid.write('%s,%s,%s\n' % (file_name, y_true_elem_cls, y_pred_elem_cls))
