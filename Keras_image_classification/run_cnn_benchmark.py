from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.python.keras.metrics import Metric
#from keras import backend as K

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
from tensorflow.keras.callbacks import Callback
from PIL import Image
import os
from os.path import join as pjoin
import numpy as np
import cv2
import shutil
from pprint import pprint
from time import time

from show_misclassified import show_misclassified
from concat_all_results import concat_all_results
from helper_keras import get_cnn_model, get_preprocess_input, get_image_size
from helper_keras import get_class_list_from_generator, make_generators
from helper_keras import get_all_metrics, save_result, plot_accuracy_and_loss

#from losses import categorical_focal_loss

batch_size = 4

#tf.keras.backend.set_floatx('float16')

def run_cnn(dataset_path, dataset_name, model_name, is_train):
    train_dir = pjoin(dataset_path, dataset_name, 'train')
    val_dir = pjoin(dataset_path, dataset_name, 'val')
    test_dir = pjoin(dataset_path, dataset_name, 'test')
#    test_dir = pjoin(dataset_path, dataset_name, 'test_BP')
#    test_dir = pjoin(dataset_path, dataset_name, 'test_Chevron')
#    test_dir = pjoin(dataset_path, dataset_name, 'test_Total')

    image_size = get_image_size(model_name)
    IMG_SHAPE = (image_size, image_size, 3)
    preprocess_input = get_preprocess_input(model_name)

    (train_generator, val_generator, test_generator) = make_generators(
        train_dir, val_dir, test_dir,
        preprocess_input, image_size, batch_size)
    test_generator.class_indices = train_generator.class_indices

    (cls_list, n_cls) = get_class_list_from_generator(train_generator)
    base_model = get_cnn_model(model_name, IMG_SHAPE)
    model_output_name = '%s_best.h5' % (model_name)
    base_model.trainable = False
    model = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(n_cls, activation='softmax')
    ])
    class_weight = {
        0: 100.,
        1: 100.,
        2: 100.,
        3: 100.,
        4: 1.,
    }
#    set_trace()
#    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
#    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
#    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = val_generator.n // batch_size
    if is_train == 1:
        steps_per_epoch = train_generator.n // batch_size
        validation_steps = val_generator.n // batch_size

        base_model.trainable = True
        print("Number of layers in the base model: ", len(base_model.layers))
        fine_tune_at = 0
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

        callbacks = [
            # keras.callbacks.ReduceLROnPlateau(),
            tf.keras.callbacks.ModelCheckpoint(
                model_output_name,
#                save_weights_only=True,
                save_best_only=True,
                monitor='val_loss',
                mode='auto',
                ),
        ]

#        model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
        model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-6),
#        model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001),
#        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
                      loss='categorical_crossentropy',
#                      loss=categorical_focal_loss(alpha=[[.25, .25, .25, .25]], gamma=2),
                      metrics=['accuracy'],
        )
        epochs = 100
        history_fine = model.fit_generator(train_generator,
                                           steps_per_epoch = steps_per_epoch,
                                           epochs=epochs,
                                           workers=4,
                                           callbacks=callbacks,
                                           validation_data=val_generator,
                                           validation_steps=validation_steps,
#                                           class_weight=class_weight,
                                           )

        # Save accuracy and F1 score
        acc_plot_file_name = '%s_%s_acc.jpg' % (dataset_name, model_name)
        loss_plot_file_name = '%s_%s_loss.jpg' % (dataset_name, model_name)
        plot_accuracy_and_loss(history_fine, acc_plot_file_name, loss_plot_file_name)
    else:
#        base_model.trainable = True
#        print("Number of layers in the base model: ", len(base_model.layers))
#        # Fine tune from this layer onwards
#        fine_tune_at = 0
#        # Freeze all the layers before the `fine_tune_at` layer
#        for layer in base_model.layers[:fine_tune_at]:
#            layer.trainable =  False
#
#        model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
##                      loss='categorical_crossentropy',
##                      metrics=['accuracy', 'Precision', 'Recall']
#                    loss=categorical_focal_loss(alpha=[[.25, .25, .25, .25, .25]], gamma=2),
#                    metrics=['accuracy'],
#                    )
        pass

    # Test
    model.load_weights(model_output_name)
#    model = tf.keras.models.load_model(model_output_name, compile=False)
    t1 = time()

    n_img = test_generator.n # Correct even if batch_size not equal to 1
    test_file_list = test_generator.filenames
    y_true = np.zeros(n_img, dtype=np.int32)
    y_pred = np.zeros(n_img, dtype=np.int32)

    batches = 0
    img_idx = 0
    for x_batch, y_batch in test_generator:
        y_true_inst = np.argmax(y_batch, axis=1)[0] # If batch_size not 1, change this
        y_true[img_idx] = y_true_inst
        pred = model.predict(x_batch)
        y_pred_inst = np.argmax(pred)
        y_pred[img_idx] = y_pred_inst

        img_idx += 1
        batches += 1
        if batches >= test_generator.n: # Only for batch_size == 1! 
            break
#    set_trace()
    save_result(dataset_name, model_name, y_true, y_pred, cls_list, test_file_list, "./")

    pred_file_name = '%s_%s_predictions.txt' % (dataset_name, model_name)
    output_folder = './%s_%s_misclassified' % (dataset_name, model_name)

    t2 = time()
    print("Running model: %s on dataset: %s takes: %f seconds" % (model_name, dataset_name, t2-t1))

#    show_misclassified(test_dir, model_name,
#        pred_file_name, output_folder)


if __name__ == "__main__":
#    dataset_path = '/home/ubuntu/TRUONG/datasets/'
#    dataset_path = '/home/eyad/Desktop/TRUONG/datasets/'
    dataset_path = '/home/researcher/Desktop/TRUONG/datasets'
#    dataset_path = '/home/eyad/Desktop/TRUONG/datasets/BALANCED_DATASETS'
#    dataset_path = '/home/eyad/Desktop/TRUONG/code/SCB+Gamma+BrightnessContrast'
#    dataset_list = ['MLC2008_train_val_test']
#    dataset_list = ['Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['Saibok_Total_augment_train_val_test']
#    dataset_list = ['Saibok_Total_balanced_binary_train_val_test']
#    dataset_list = ['Saibok_Total_Chevron_merge_binary_train_val_test']
#    dataset_list = [
#        'Saibok_Total_all_10_percent_train_val_test',
#        'Saibok_Total_all_20_percent_train_val_test',
#        'Saibok_Total_all_30_percent_train_val_test',
#        'Saibok_Total_all_40_percent_train_val_test',
#        'Saibok_Total_all_50_percent_train_val_test',
#    ]
#    dataset_list = ['Saibok_Total_Chevron_BP_merge_binary_train_val_test']
#    dataset_list = ["Saibok_Total_balanced_binary_dataset"]
#    dataset_list = ["Total_Chevron_BP_balanced_binary_merged"]
#    dataset_list = ["Saibok_BP_balanced_dataset_GammaContrastAdjustment"]
#    dataset_list = ["Saibok_BP_balanced_binary_dataset_GammaContrastAdjustment"]
#    dataset_list = ["Saibok_BP_MLR18_2021_train_val_test_Gamma_ContrastAdjustment"]
#    dataset_list = ["Saibok_Chevron_balanced_dataset_overlay_removal_Gamma_Contrast_Adjustment"]
#    dataset_list = ["Total_Chevron_BP_balanced_Gamma_ContrastAdjustment"]
#    dataset_list = ["Saibok_Total_train_val_test_fixed"]
#    dataset_list = ['BP_binary_train_val_test']
#    dataset_list = ['Saibok_BP_balanced_binary_dataset']
#    dataset_list = ['Saibok_Chevron_balanced_binary_dataset']
    dataset_list = ['DatasetA_overlay_removed']
#    dataset_list = ['Saibok_BP_balanced_dataset']
#    dataset_list = ["Saibok_Total_Chevron_BP_merge_train_val_test"]
#    dataset_list = ['Saibok_Chevron_data_binary_classify_train_val_test']
#    dataset_list = ['Saibok_Chevron_data_train_val_test']
#    dataset_list = ['Saibok_2014_2017_oversampled_train_val_test']
#    dataset_list = ['FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['FusionEnhanced_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['RemovedBackscatter_Saibok_2014_2017_balanced_train_val_test']
    model_list = ['VGG16', 'Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
#    model_list = ['InceptionV3', 'DenseNet121', 'XCeption']
#    model_list = ['InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
#    model_list = ['InceptionV3', 'XCeption']
#    model_list = ['VGG16']
#    model_list = ['DenseNet121', 'InceptionV3', 'XCeption']
#    model_list = ["DenseNet121"]
#    model_list = ['Resnet50']
#    model_list = ['InceptionV3']
    is_train = 1
#    is_train = 0
    for dataset_name in dataset_list:
        for model_name in model_list:
            print("MODEL NAME: %s, dataset name: %s" % (model_name, dataset_name))
            run_cnn(dataset_path, dataset_name, model_name, is_train)

#        concat_all_results(dataset_name, model_list)


