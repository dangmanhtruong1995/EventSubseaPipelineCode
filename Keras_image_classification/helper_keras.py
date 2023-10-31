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

#plt.rcParams.update({'font.size': 40})

np.seterr(all='raise')

class SnapshotSaver(keras.callbacks.Callback):
    """Custom callback to save model every n_epoch_per_snapshot epochs."""
    def __init__(self, model_name, n_epoch_per_snapshot, output_path):
        super().__init__()
        self.model_name = model_name
        self.n_epoch_per_snapshot = n_epoch_per_snapshot
        self.output_path = output_path

    def on_epoch_end(self, epoch, logs={}):
        model_name = self.model_name
        n_epoch_per_snapshot = self.n_epoch_per_snapshot
        output_path = self.output_path
        if (epoch % n_epoch_per_snapshot == 0) and (epoch > 0):
            print("Saving at epoch %d" % (epoch))
            self.model.save(
                pjoin(output_path, "model_%s_epoch_%d.h5" % (model_name, epoch)))

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
        raise Exception('In run_cnn: Model name %s not supported!' % (model_name))
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

def make_generators(train_dir, val_dir, test_dir, preprocess_input,
         image_size, batch_size):
    """Function to make train, validation and test generators """
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
#                                   brightness_range=[0.5, 1.5])
    )
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )

    train_generator = train_datagen.flow_from_directory(
                    train_dir,  # Source directory for the training images
                    target_size=(image_size, image_size),
                    shuffle=True,
                    batch_size=batch_size,
                    seed=0,
                    )
    val_generator = train_datagen.flow_from_directory(
                    val_dir,  # Source directory for the validation images
                    target_size=(image_size, image_size),
                    shuffle=False,
                    batch_size=batch_size,
                    seed=0,
                    )
    test_generator = test_datagen.flow_from_directory(
                    test_dir, # Source directory for the test images
                    target_size=(image_size, image_size),
                    shuffle=False,
                    batch_size=1,
                    seed=0,
                    )
    return (train_generator, val_generator, test_generator)


def plot_accuracy_and_loss(history_fine, acc_plot_file_name, loss_plot_file_name):
    acc = history_fine.history['accuracy']
    val_acc = history_fine.history['val_accuracy']
    loss = history_fine.history['loss']
    val_loss = history_fine.history['val_loss']
    plt.plot(acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
    plt.savefig(acc_plot_file_name)
    plt.close()

    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
    plt.savefig(loss_plot_file_name)
    plt.close()


def get_class_list_from_generator(the_generator):
    n_cls = len(the_generator.class_indices.keys())
    cls_list = []
    for cls_idx in range(n_cls):
        for cls_name in the_generator.class_indices.keys():
            if the_generator.class_indices[cls_name] == cls_idx:
                cls_list.append(cls_name)
    return (cls_list, n_cls)


def get_precision_recall_f1_list_from_confusion_matrix(conf):
    """ Get precision, recall and F1 for each class from multi-class confusion matrix
    This function is useful if your classes in y_true and y_pred have discrepancies
    Assumes that conf[i, j] is number of instances which is supposed to be in
    class i but predicted in class j

    """
    n_cls = conf.shape[0]
    prec_list = np.zeros(n_cls)
    rec_list = np.zeros(n_cls)
    f1_list = np.zeros(n_cls)

    for cls_idx in range(n_cls):
        # Calculate TP, FP, FN, TN first
        TP = conf[cls_idx, cls_idx]
        FP = 0
        FN = 0
        TN = 0
        for i1 in range(n_cls):
            for j1 in range(n_cls):
                if (i1 != cls_idx) and (j1 == cls_idx):
                    FP += conf[i1, j1]
                elif (i1 == cls_idx) and (j1 != cls_idx):
                    FN += conf[i1, j1]
                elif (i1 != cls_idx) and (j1 != cls_idx):
                    TN += conf[i1, j1]
                else:
                    pass

        # Then calculate precision, recall and F1
        try:
            prec = TP/(1.0*(TP+FP))
        except:
#            set_trace()
            prec = 0

        try:
            rec = TP/(1.0*(TP+FN))
#            set_trace()
        except:
            rec = 0

        try:
            f1 = (2*prec*rec)/(1.0*(prec+rec))
        except:
            f1 = 0

#        set_trace()

        prec_list[cls_idx] = prec
        rec_list[cls_idx] = rec
        f1_list[cls_idx] = f1

    return (prec_list, rec_list, f1_list)

def get_all_metrics(y_true, y_pred, n_cls):
    """ Function to get all the metrics (Accuracy, F1 score, confusion matrix, ...)
    Needs n_cls because y_pred and y_true might have discrepancies
    """

    n_row = np.size(y_pred)
    conf = np.zeros((n_cls, n_cls))
    for idx in range(n_row):
        conf[y_true[idx], y_pred[idx]] += 1

    accuracy = np.trace(conf) / (1.0*(np.sum(conf)))
    (prec_list, rec_list, f1_list) = get_precision_recall_f1_list_from_confusion_matrix(conf)
    prec_avg = np.sum(prec_list) / (1.0*n_cls)
    rec_avg = np.sum(rec_list) / (1.0*n_cls)
    f1_avg = np.sum(f1_list) / (1.0*n_cls)

    return (conf, accuracy, prec_avg, rec_avg, f1_avg, prec_list, rec_list, f1_list)

def save_result(dataset_name, model_name, y_true, y_pred, cls_list, test_file_list, output_path):
    n_cls = len(cls_list)
    (conf, accuracy, prec_avg, rec_avg, f1_avg, prec_list, rec_list, f1_list) = get_all_metrics(y_true, y_pred, n_cls)

    np.savetxt(pjoin(output_path, '%s_%s_conf.csv' % (dataset_name, model_name)),
         conf, fmt='%d', delimiter=',')
    fig, ax = plt.subplots()
 #   fig.set_figwidth(50)
 #   fig.set_figheight(50)  
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, 
                               display_labels=cls_list)
#    set_trace()
#    font = {
#        #'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 60,
#    }
#    plt.rc('font', **font)
#    plt.xlabel('xlabel', fontsize=40)
#    plt.rc('font', **font)
#    plt.xlabel('xlabel', fontsize=40
#    plt.tight_layout(pad=0)
    disp.plot(ax=ax)
    plt.show()
#    plt.savefig(pjoin(output_path, '%s_%s_conf_visualize.png' % (dataset_name, model_name)), bbox_inches="tight")
    plt.savefig(pjoin(output_path, '%s_%s_conf_visualize.png' % (dataset_name, model_name)))
    plt.close()

    with open(pjoin(output_path, '%s_%s_result.csv' % (dataset_name, model_name)), 'wt') as fid:
        fid.write('Acc,F1\n')
        fid.write('%.5f,%.5f\n' % (accuracy, f1_avg))

    with open(pjoin(output_path, '%s_%s_result_ALL.csv' % (dataset_name, model_name)), 'wt') as fid:
        fid.write('Acc,Precision,Recall,F1\n')
        fid.write('%.5f,%.5f,%.5f,%.5f\n' % (accuracy, prec_avg, rec_avg, f1_avg))

        fid.write("Precision list\n")
        for cls_idx in range(n_cls):
            fid.write("%.5f," % (prec_list[cls_idx]))
        fid.write("\n")

        fid.write("Recall list\n")
        for cls_idx in range(n_cls):
            fid.write("%.5f," % (rec_list[cls_idx]))
        fid.write("\n")

        fid.write("F1 list\n")
        for cls_idx in range(n_cls):
            fid.write("%.5f," % (f1_list[cls_idx]))
        fid.write("\n")

    with open(pjoin(output_path, '%s_%s_predictions.txt' % (dataset_name, model_name)), 'wt') as fid:
        for (file_name, y_true_elem, y_pred_elem) in zip(test_file_list, y_true, y_pred):
            y_true_elem_cls = cls_list[y_true_elem]
            y_pred_elem_cls = cls_list[y_pred_elem]
            fid.write('%s,%s,%s\n' % (file_name, y_true_elem_cls, y_pred_elem_cls))

def main():
    conf = np.array(
        [
            [10,  5,  4],
            [2,  20,  3],
            [1,   3, 30],
        ])
    (prec_list, rec_list, f1_list) = get_precision_recall_f1_list_from_confusion_matrix(conf)
    print("Confusion matrix:")
    print(conf)
    print("Precision for each class:")
    print(prec_list)
    print("Recall for each class:")
    print(rec_list)
    print("F1 for each class:")
    print(f1_list)

if __name__ == "__main__":
    main()

