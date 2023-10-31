import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import sys
from pdb import set_trace
import numpy as np
from shutil import copyfile, rmtree
import os
from os.path import join as pjoin
import json
import math
from random import shuffle
import cv2

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2

from helper_keras import get_cnn_model, get_preprocess_input, get_image_size
from helper_keras import save_result, make_generators

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)



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

class KerasCNNModelWrapper:
    def __init__(self, model_name, cls_list):
        image_size = get_image_size(model_name)
        img_shape = (image_size, image_size, 3)

        n_cls = len(cls_list)

        base_model = get_cnn_model(model_name, img_shape)
        base_model.trainable = False
        model = tf.keras.Sequential([
          base_model,
          keras.layers.GlobalAveragePooling2D(),
#          keras.layers.Dense(n_cls, activation='softmax', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
          keras.layers.Dense(n_cls, activation='softmax')
        ])
        model.compile(
#            optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
            optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
#            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss='categorical_crossentropy',
#            metrics=['accuracy', 'Precision', 'Recall'])
            metrics=['accuracy'])

        self.image_size = image_size
        self.img_shape = img_shape
        self.model = model
        self.base_model = base_model
        self.model_name = model_name
        self.n_cls = n_cls
        self.cls_list = cls_list

    def load(self, model_path):
        self.model.load_weights(model_path)

    def train(self, train_generator, val_generator, epochs, batch_size, prefix_tag=""):
        image_size = self.image_size
        img_shape = self.img_shape
        model = self.model
        model_name = self.model_name
        n_cls = self.n_cls
        cls_list = self.cls_list

        model_output_name = '%s_%s_best.h5' % (prefix_tag, model_name)
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_output_name,
                save_weights_only=True,
                save_best_only=True,
                mode='auto',
                ),
        ]

        # Train
#        self.base_model.trainable = False
#        steps_per_epoch = train_generator.n // batch_size
#        validation_steps = val_generator.n // batch_size
#        history_fine = model.fit_generator(train_generator,
#            steps_per_epoch=steps_per_epoch,
#            epochs=10,
#            workers=4,
#            callbacks=callbacks,
#            validation_data=val_generator,
#            validation_steps=validation_steps,
#        )

#        self.base_model.trainable = True
        steps_per_epoch = train_generator.n // batch_size
        validation_steps = val_generator.n // batch_size
        history_fine = model.fit_generator(train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            workers=4,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=validation_steps,
        )
        model.load_weights(model_output_name)

        # Plot accuracy and loss
        acc = history_fine.history['accuracy']
        val_acc = history_fine.history['val_accuracy']
        loss = history_fine.history['loss']
        val_loss = history_fine.history['val_loss']

        plt.plot(acc, label='Training accuracy')
        plt.plot(val_acc, label='Validation accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.show()
        plt.savefig('%s_%s_acc.jpg' % (prefix_tag, model_name))
        plt.close()

        plt.plot(loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.legend()
        plt.title('Loss')
        plt.show()
        plt.savefig('%s_%s_loss.jpg' % (prefix_tag, model_name))
        plt.close()





        self.model = model

    def predict_proba(self, test_generator):
        y_prob = self.model.predict_generator(test_generator, test_generator.n)
        return y_prob

    def predict(self, test_generator):
        y_prob = self.model.predict_generator(test_generator, test_generator.n)
        y_pred = np.argmax(Y_pred, axis=1)
        return y_pred

def create_metadata(dataset_name, model_name, epochs, batch_size,
        train_path, val_path, test_path, prefix_tag, output_path_dataset):
    preprocess_input = get_preprocess_input(model_name)
    image_size = get_image_size(model_name)
    train_generator, val_generator, test_generator, cls_list = make_generators(train_path, val_path, test_path, preprocess_input,
         image_size, batch_size)

    # Train, predict the metadata then save
    model = KerasCNNModelWrapper(model_name, cls_list)
    model.train(train_generator, val_generator, epochs, batch_size, prefix_tag=prefix_tag)
    y_prob = model.predict_proba(test_generator)
    y_pred = np.argmax(y_prob, axis=1)

    # Save
    file_list = test_generator.filenames
    n_cls = len(cls_list)
    output_file_name = "metadata_%s_%s.txt" % (model_name, prefix_tag)
    output_path = pjoin(output_path_dataset, output_file_name)
    with open(output_path, "wt") as fid:
        for file_idx, file_name in enumerate(file_list):
            fid.write("%s," % (file_name))
            for cls_idx in range(n_cls):
                if cls_idx < (n_cls-1):
                    fid.write("%.6f," % y_prob[file_idx, cls_idx])
                else:
                    fid.write("%.6f" % y_prob[file_idx, cls_idx])
            fid.write("\n")

    # If test then save the metadata of validation set as well
    if "Test" in prefix_tag:
        y_prob = model.predict_proba(val_generator)
        y_pred = np.argmax(y_prob, axis=1)

        file_list = val_generator.filenames
        n_cls = len(cls_list)
        output_file_name = "metadata_%s_%s.txt" % (model_name, "val")
        output_path = pjoin(output_path_dataset, output_file_name)
        with open(output_path, "wt") as fid:
            for file_idx, file_name in enumerate(file_list):
                fid.write("%s," % (file_name))
                for cls_idx in range(n_cls):
                    if cls_idx < (n_cls-1):
                        fid.write("%.6f," % y_prob[file_idx, cls_idx])
                    else:
                        fid.write("%.6f" % y_prob[file_idx, cls_idx])
                fid.write("\n")


def create_metadata_at_fold(dataset_path, dataset_name, model_name, fold_idx, epochs, batch_size, output_path_dataset):
    print("Creating metadata at fold: %d" % (fold_idx+1))

    train_path = pjoin(dataset_path, dataset_name, "train_fold_%d" % (fold_idx+1), "train")
    val_path = pjoin(dataset_path, dataset_name, "val")
    test_path = pjoin(dataset_path, dataset_name, "train_fold_%d" % (fold_idx+1), "val")

    prefix_tag = "Fold_%d" % (fold_idx+1)

    create_metadata(dataset_name, model_name, epochs, batch_size,
        train_path, val_path, test_path, prefix_tag, output_path_dataset)

def create_metadata_test(dataset_path, dataset_name, model_name, epochs, batch_size, output_path_dataset):
    print("Creating metadata for test set")

    train_path = pjoin(dataset_path, dataset_name, "train")
    val_path = pjoin(dataset_path, dataset_name, "val")
    test_path = pjoin(dataset_path, dataset_name, "test")

    prefix_tag = "Test"

    create_metadata(dataset_name, model_name, epochs, batch_size,
        train_path, val_path, test_path, prefix_tag, output_path_dataset)

def main():
    dataset_path = '/home/truong/Desktop/TRUONG/datasets'

#    dataset_list = ['MLC2008_train_val_test']
#    dataset_list = ['Saibok_Total_balanced']
#    dataset_list = ['Saibok_Chevron_dataset']
#    dataset_list = ['Saibok_BP_balanced_dataset']
    dataset_list = ['Saibok_Chevron_dataset', 'Saibok_BP_balanced_dataset']
#    dataset_list = ['imbalanced_mnist_example']
#    dataset_list = ['MNIST']
#    dataset_list = ['Saibok_2014_2017_oversampled_train_val_test']
#    dataset_list = ['FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['Saibok_2014_2017_balanced_2000_train_val_test']
#    dataset_list = ['FusionEnhanced_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['RemovedBackscatter_Saibok_2014_2017_balanced_train_val_test']

    model_list = ['VGG16', 'Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
#    model_list = ['InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'Xception']
#    model_list = ['XCeption']
#    model_list = ['VGG16']
#    model_list = ['DenseNet121']

#    model_list = ['Resnet50', "VGG16"]
#    model_list = ['VGG16']
#    model_list = ['Resnet50']
    is_train = 1
    fold_idx_list = [0, 1, 2, 3, 4] # Starts from 0
#    fold_idx_list = [0] # Starts from 0
#    fold_idx_list = []
    epochs = 100
    batch_size = 16
    output_path_base = './RESULT'

    for dataset_name in dataset_list:
        output_path_dataset = pjoin(output_path_base, dataset_name)
        if os.path.exists(output_path_dataset) is False:
            mkdir(output_path_dataset)
        for model_name in model_list:
            print("Dataset: %s. Model name: %s" % (dataset_name, model_name))
            for fold_idx in fold_idx_list:
                create_metadata_at_fold(dataset_path, dataset_name, model_name, fold_idx, epochs, batch_size, output_path_dataset)

            create_metadata_test(dataset_path, dataset_name, model_name, epochs, batch_size, output_path_dataset)


    pass

if __name__ == "__main__":
    main()
