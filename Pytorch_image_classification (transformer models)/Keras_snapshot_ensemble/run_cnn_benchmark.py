from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.python.keras.metrics import Metric
from keras import backend as K

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

#from show_misclassified import show_misclassified
#from concat_all_results import concat_all_results

batch_size = 16

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


def run_cnn(dataset_path, dataset_name, model_name, is_train):
    image_size = get_image_size(model_name)
    IMG_SHAPE = (image_size, image_size, 3)

    train_dir = pjoin(dataset_path, dataset_name, 'train')
    val_dir = pjoin(dataset_path, dataset_name, 'val')
    test_dir = pjoin(dataset_path, dataset_name, 'test')
#    test_dir = '/home/ubuntu/TRUONG/datasets/Saibok_Chevron_data_all_1'
    # train_dir = '/truong-rerun/Run_CNN_benchmark/data/temp_Inception'

    label_list = os.listdir(train_dir)
    num_classes = len(label_list)
    preprocess_input = get_preprocess_input(model_name)

    # Convert train1 and validation dir into a train dir
#    shutil.rmtree(train_dir)
#    os.mkdir(train_dir)
#    label_list = os.listdir(train_dir)
#    num_classes = len(label_list)
#    for label in label_list:
#        os.mkdir(pjoin(train_dir, label))
#        for file_name in os.listdir(pjoin(train_dir, label)):
#            shutil.copyfile(pjoin(train_dir, label, file_name), pjoin(train_dir, label, file_name[:-4] + "_train1.jpg"))
#        for file_name in os.listdir(pjoin(val_dir, label)):
#            shutil.copyfile(pjoin(val_dir, label, file_name), pjoin(train_dir, label, file_name[:-4] + "_val.jpg"))

    # Rescale all images by 1./255 and apply image augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
#                                   rescale=1./255,
                                   preprocessing_function=preprocess_input,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.5])
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
#        rescale=1./255,
        preprocessing_function=preprocess_input,
        )
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
#        rescale=1./255,
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

    base_model = get_cnn_model(model_name, IMG_SHAPE)
    model_output_name = '%s_best.h5' % (model_name)

    base_model.trainable = False
    model = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall'])
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = val_generator.n // batch_size
    if is_train == 1:
        epochs = 10
        steps_per_epoch = train_generator.n // batch_size
        validation_steps = val_generator.n // batch_size

        callbacks = [
            # keras.callbacks.ReduceLROnPlateau(),
            keras.callbacks.ModelCheckpoint(
                model_output_name,
                save_weights_only=True,
                save_best_only=True,
                mode='auto',
                ),
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

#        model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
        model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'Precision', 'Recall'])
        epochs = 100
        history_fine = model.fit_generator(train_generator,
                                           steps_per_epoch = steps_per_epoch,
                                           epochs=epochs,
                                           workers=4,
                                           callbacks=callbacks,
                                           validation_data=test_generator,
                                           validation_steps=validation_steps,
                                           )

        # Save accuracy and F1 score
        acc = history_fine.history['acc']
        val_acc = history_fine.history['val_acc']
        loss = history_fine.history['loss']
        val_loss = history_fine.history['val_loss']
        plt.plot(acc, label='Training accuracy')
        plt.plot(val_acc, label='Validation accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.show()
        plt.savefig('%s_%s_acc.jpg' % (dataset_name, model_name))
        plt.close()

        plt.plot(loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.legend()
        plt.title('Loss')
        plt.show()
        plt.savefig('%s_%s_loss.jpg' % (dataset_name, model_name))
        plt.close()
    else:
        base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))
        # Fine tune from this layer onwards
        fine_tune_at = 0
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

        model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=2e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy', 'Precision', 'Recall']
                    )

    model.load_weights(model_output_name)

    # Test
    label_list = train_generator.class_indices
    numeric_to_class = {}
    for key, val in label_list.items():
        numeric_to_class[val] = key
#    total_num_images = len(test_generator)
    total_num_images = 0
    acc_num_images = 0

    score = model.evaluate_generator(generator=test_generator,               # Generator yielding tuples
#                                     steps=generator.samples//nBatches, # number of steps (batches of samples) to yield from generator before stopping
                                     steps=test_generator.n,
                                     max_queue_size=10,                 # maximum size for the generator queue
                                     workers=1,                         # maximum number of processes to spin up when using process based threading
                                     use_multiprocessing=False,         # whether to use process-based threading
                                     verbose=0)
    loss_val = score[0]
    acc_val = score[1]
#    precision_val = score[2]
#    recall_val = score[3]
#    f1_val = 2*((precision_val*recall_val)/(precision_val+recall_val+K.epsilon()))
##    set_trace()
#    print("Loss: %.3f, acc: %.3f, precision: %.3f, recall: %.3f, F1: %.3f" % (loss_val, acc_val, precision_val, recall_val, f1_val))
#    with open("%s_%s_loss_acc_evaluate_generator.txt" % (dataset_name, model_name), "wt") as fid:
#        fid.write("Loss: %.3f, acc: %.3f, precision: %.3f, recall: %.3f, F1: %.3f" % (loss_val, acc_val, precision_val, recall_val, f1_val))
#        fid.write("\n")

    test_generator.reset()

    Y_pred = model.predict_generator(test_generator, test_generator.n)
    y_pred = np.argmax(Y_pred, axis=1)
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
        pass
#    set_trace()
    np.savetxt('%s_%s_conf.csv' % (dataset_name, model_name), conf, fmt='%d', delimiter=',')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                               display_labels=cls_list)
    disp.plot()
    plt.show()
    plt.savefig('%s_%s_conf_visualize.png' % (dataset_name, model_name))
    plt.close()

    with open('%s_%s_result.csv' % (dataset_name, model_name), 'wt') as fid:
        fid.write('Acc,F1\n')
        fid.write('%.5f,%.5f\n' % (acc, f1))

    test_file_list = test_generator.filenames
    with open('%s_%s_predictions.txt' % (dataset_name, model_name), 'wt') as fid:
        for (file_name, y_true_elem, y_pred_elem) in zip(test_file_list, y_true, y_pred):
            y_true_elem_cls = cls_list[y_true_elem]
            y_pred_elem_cls = cls_list[y_pred_elem]
            fid.write('%s,%s,%s\n' % (file_name, y_true_elem_cls, y_pred_elem_cls))
#            set_trace()


#    print(classification_report(
#        test_generator.classes,
#        y_pred,
#        target_names=test_generator.class_indices.keys()))

#    test_generator.reset()
#    while True:
#        elem = test_generator.next()
#        set_trace()
#
#
#    pred_list = np.zeros(total_num_images)
#    gt_list = np.zeros(total_num_images)
#    gt_idx = 0
#    with open("%s_prediction_%s.txt" % (dataset_name, model_name), "wt") as fid:
#        fid.write("Label list:\n")
#        for label in label_list:
#            fid.write("%s," % label)
#        fid.write("\n")
#        fid.write("true_class,predicted_class\n")
#        fid.write("--------------------------\n")
#        for label_idx, label in enumerate(label_list):
#            testing_dir = pjoin(test_dir, label)
#            for img_file in os.listdir(testing_dir):
#                img_fullpath = pjoin(testing_dir, img_file)
#
#                img = keras.preprocessing.image.load_img(
#                    img_fullpath, target_size=(image_size, image_size)
#                )
#                img_array = keras.preprocessing.image.img_to_array(img)
#                img_array = np.expand_dims(img_array, 0)  # Create batch axis
#                img_array = preprocess_input(img_array)
#
#                pred_class_num = model.predict_classes(img_array)
#                pred_class_num = pred_class_num[0]
#                true_class_num = label_list[label]
#                predicted_label = numeric_to_class[pred_class_num]
##                set_trace()
##                img = cv2.imread(pjoin(testing_dir, img_file))
##                img_resized = cv2.resize(img, (image_size, image_size), interpolation = cv2.INTER_AREA)
##                img1 = np.reshape(img_resized, (1, img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
##                pred_class_num = model.predict_classes(img1)
##                pred_class_num = pred_class_num[0]
##                true_class_num = label_list[label]
##                predicted_label = numeric_to_class[pred_class_num]
#
#                fid.write("%s,%s\n" % (label, predicted_label))
#                if predicted_label == label:
#                    acc_num_images += 1
#                gt_idx += 1
#                total_num_images += 1


if __name__ == "__main__":
    dataset_path = "/home/truong/Desktop/TRUONG/datasets"

#    dataset_list = ['MLC2008_train_val_test']
#    dataset_list = ['Saibok_2014_2017_balanced_train_val_test']
    dataset_list = ['MNIST']
#    dataset_list = ['Saibok_2014_2017_oversampled_train_val_test']
#    dataset_list = ['FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['Saibok_2014_2017_balanced_2000_train_val_test']
#    dataset_list = ['FusionEnhanced_Saibok_2014_2017_balanced_train_val_test']
#    dataset_list = ['RemovedBackscatter_Saibok_2014_2017_balanced_train_val_test']
#    model_list = ['VGG16', 'Resnet50', 'InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'XCeption']
#    model_list = ['InceptionV3', 'InceptionResNetV2', 'DenseNet121', 'Xception']
#    model_list = ['XCeption']
#    model_list = ['VGG16']
#    model_list = ['DenseNet121']
    model_list = ['Resnet50']
    is_train = 1
    for dataset_name in dataset_list:
        for model_name in model_list:
            print("MODEL NAME: %s, dataset name: %s" % (model_name, dataset_name))
            run_cnn(dataset_path, dataset_name, model_name, is_train)

            pred_file_name = '%s_%s_predictions.txt' % (dataset_name, model_name)
            output_folder = './%s_%s_misclassified' % (dataset_name, model_name)
#            show_misclassified(dataset_path, dataset_name, model_name,
#                pred_file_name, output_folder)

#        concat_all_results(dataset_name, model_list)


