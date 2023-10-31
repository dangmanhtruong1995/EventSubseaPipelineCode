from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.python.keras.metrics import Metric
from keras import backend as K
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
from helper_keras import get_cnn_model, get_preprocess_input, get_image_size
from helper_keras import save_result, make_generators

batch_size = 4

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

def make_snapshots(dataset_path, dataset_name, model_name,
        n_snapshot, n_epoch_per_snapshot, output_path_base):
    image_size = get_image_size(model_name)
    IMG_SHAPE = (image_size, image_size, 3)

    train_dir = pjoin(dataset_path, dataset_name, 'train')
    val_dir = pjoin(dataset_path, dataset_name, 'val')
    test_dir = pjoin(dataset_path, dataset_name, 'test')

#    label_list = os.listdir(train_dir)
#    n_cls = len(label_list)
    preprocess_input = get_preprocess_input(model_name)
    (train_generator, val_generator, test_generator, label_list) = make_generators(
        train_dir, val_dir, test_dir,
        preprocess_input, image_size, batch_size)
    n_cls = len(label_list)

    base_model = get_cnn_model(model_name, IMG_SHAPE)
    base_model.trainable = False
    model = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(n_cls, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = val_generator.n // batch_size
    epochs = 2

    callbacks = [
    ]

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  workers=4,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=validation_steps)

    # After top classifier is trained, we finetune the layers of the network
    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine tune from this layer onwards
    fine_tune_at = 0
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    epochs = (n_snapshot+1) * n_epoch_per_snapshot
    callbacks = [
        SnapshotSaver(model_name, n_epoch_per_snapshot, output_path_base),
    ]

    history_fine = model.fit_generator(train_generator,
                                       steps_per_epoch = steps_per_epoch,
                                       epochs=epochs,
                                       workers=4,
                                       callbacks=callbacks,
                                       validation_data=test_generator,
                                       validation_steps=validation_steps,
                                       )

    # Save accuracy and F1 score
    acc = history_fine.history['accuracy']
    val_acc = history_fine.history['val_accuracy']
    loss = history_fine.history['loss']
    val_loss = history_fine.history['val_loss']
    plt.plot(acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()
    plt.savefig(pjoin(output_path_base, '%s_%s_acc.jpg' % (dataset_name, model_name)))
    plt.close()

    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
    plt.savefig(pjoin(output_path_base, '%s_%s_loss.jpg' % (dataset_name, model_name)))
    plt.close()

def snapshot_sumrule_ensemble(dataset_path, dataset_name, model_name,
        n_snapshot, n_epoch_per_snapshot, output_path_base):
    image_size = get_image_size(model_name)
    IMG_SHAPE = (image_size, image_size, 3)

    train_dir = pjoin(dataset_path, dataset_name, 'train')
    val_dir = pjoin(dataset_path, dataset_name, 'val')
    test_dir = pjoin(dataset_path, dataset_name, 'test')

    label_list = os.listdir(train_dir)
    n_cls = len(label_list)
    preprocess_input = get_preprocess_input(model_name)

    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    test_generator = test_datagen.flow_from_directory(
                    test_dir, # Source directory for the test images
                    target_size=(image_size, image_size),
                    shuffle=False,
                    batch_size=1,
                    seed=0,
                    )

    snapshot_epoch_list = np.arange(
        n_epoch_per_snapshot,
        (n_snapshot+1)*n_epoch_per_snapshot,
        n_epoch_per_snapshot)
    snapshot_prob_all = np.zeros((len(snapshot_epoch_list), test_generator.n, n_cls))
    for idx, snapshot_epoch_idx in enumerate(snapshot_epoch_list):
        # Load snapshot
        tf.keras.backend.clear_session()
        base_model = get_cnn_model(model_name, IMG_SHAPE)
        base_model.trainable = True
        model = tf.keras.Sequential([
          base_model,
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dense(n_cls, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'Precision', 'Recall'])
        snapshot_name = "model_%s_epoch_%d.h5" % (model_name, snapshot_epoch_idx)
        model.load_weights(
            pjoin(output_path_base, snapshot_name))

        # Evaluate snapshot model on test folder, save the results from each individual snapshot
        snapshot_result_folder = pjoin(
            output_path_base,
            'Result_%s_snapshot_epoch_%d' % (model_name, snapshot_epoch_idx))
        mkdir(snapshot_result_folder)

        test_generator.reset()
        Y_prob = model.predict_generator(test_generator, test_generator.n)
        y_pred = np.argmax(Y_prob, axis=1)

        save_result(dataset_name, model_name, y_pred, test_generator,
             snapshot_result_folder)
        snapshot_prob_all[idx, :, :] = Y_prob

    # Now perform sumrule based on snapshot results
    for idx, snapshot_epoch_idx in enumerate(snapshot_epoch_list):
        sumrule_result_folder = pjoin(
            output_path_base,
            'Result_%s_snapshot_sumrule_epoch_%d' % (model_name, snapshot_epoch_idx))
        mkdir(sumrule_result_folder)
        test_generator.reset()

        sumrule_prob = np.sum(snapshot_prob_all[:idx+1,:, :], axis=0)
        sumrule_pred = np.argmax(sumrule_prob, axis=1)
#        set_trace()

        save_result(dataset_name, model_name, sumrule_pred, test_generator,
             sumrule_result_folder)


if __name__ == "__main__":
    dataset_path = '/home/truong/Desktop/TRUONG/datasets'
#    output_path_root = './SNAPSHOT_RESULT_Saibok_BP_balanced_dataset'
    output_path_root = './SNAPSHOT_RESULT_DatasetB_overlay_removed'

#    dataset_list = ['MLC2008_train_val_test']
#    dataset_list = ['Saibok_Total_balanced']
#    dataset_list = ['Saibok_BP_balanced_dataset']
#    dataset_list = ['Saibok_Chevron_dataset']
    dataset_list = ['DatasetB_overlay_removed']
#    dataset_list = ['Saibok_GammaAdjustment']
#    dataset_list = ['Saibok_2014_2017_oversampled_train_val_test']
#    dataset_list = ['FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['Saibok_2014_2017_balanced_2000_train_val_test']
#    dataset_list = ['FusionEnhanced_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['RemovedBackscatter_Saibok_2014_2017_balanced_train_val_test']
#    model_list = ['Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
    model_list = ['VGG16', 'Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
#    model_list = ['InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'Xception']
#    model_list = ['InceptionResNetV2', 'DenseNet121', 'XCeption']
#    model_list = ['DenseNet121', 'XCeption']
#    model_list = ['InceptionResNetV2']
#    model_list = ['XCeption']
#    model_list = ['VGG16']
#    model_list = ['DenseNet121']
#    model_list = ['Resnet50']
#    model_list = ['InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
    n_snapshot = 10
    n_epoch_per_snapshot = 40
    bool_make_snapshot = True
#    bool_make_snapshot = False
    for dataset_name in dataset_list:
        for model_name in model_list:
            print("MODEL NAME: %s, dataset name: %s" % (model_name, dataset_name))
            output_path_base = pjoin(output_path_root, "%s" % (model_name))
            if bool_make_snapshot is True:
                mkdir(output_path_base)
                make_snapshots(dataset_path, dataset_name, model_name,
                    n_snapshot, n_epoch_per_snapshot, output_path_base)
            snapshot_sumrule_ensemble(dataset_path, dataset_name,
                model_name, n_snapshot, n_epoch_per_snapshot, output_path_base)




