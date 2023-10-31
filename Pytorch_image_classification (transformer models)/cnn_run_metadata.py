from __future__ import print_function 
from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import PIL
from pdb import set_trace
from os.path import join as pjoin
import time
import os
import copy
import pandas as pd

from scipy.special import softmax

sns.set_style('darkgrid')

# Detect if we have a GPU available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract,
         use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg16":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception_v3":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "SwinTransformer":
        HUB_URL = "SharanSMenon/swin-transformer-hub:main"
        MODEL_NAME = "swin_tiny_patch4_window7_224"
        # check hubconf for more models.
        model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub
        for param in model.parameters(): #freeze model
            param.requires_grad = False

        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
#        model = model.to(device)
#        input_size = n_inputs
        input_size = 224
        model_ft = model

    elif model_name == "VisionTransformer":
        HUB_URL = "facebookresearch/deit:main"
        MODEL_NAME = "deit_tiny_patch16_224"
        # check hubconf for more models.
        model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub
        for param in model.parameters(): #freeze model
            param.requires_grad = False

        n_inputs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
#        model = model.to(device)
#        input_size = n_inputs
        input_size = 224
        model_ft = model

    elif model_name == "MaxViT":
        weights = torchvision.models.MaxVit_T_Weights.DEFAULT
        model = torchvision.models.maxvit_t(weights=weights)
        model.classifier[-1] = nn.Linear(in_features=512, out_features=num_classes)
        input_size = 224
        model_ft = model

    elif model_name == "Regnet":
        weights = torchvision.models.RegNet_Y_1_6GF_Weights.DEFAULT
        model = torchvision.models.regnet_y_1_6gf(weights=weights)
        model.fc = nn.Linear(in_features=888, out_features=num_classes)
        input_size = 224
        model_ft = model

    elif model_name == "ConvNeXt":
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        model = torchvision.models.convnext_base(weights=weights)
        model.classifier[-1] = nn.Linear(in_features=1024, out_features=num_classes)
        input_size = 224
        model_ft = model

    elif model_name == "EfficientNetV2":
        weights = torchvision.models.EfficientNet_V2_M_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_m(weights=weights)
#        set_trace()
        model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)
        input_size = 224
        model_ft = model

    elif model_name == "ReXNet":
        model = timm.create_model('rexnet_100', pretrained=True, num_classes=num_classes)
        input_size = 224
        model_ft = model

    elif model_name == "NoisyStudent":
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=num_classes)
        input_size = 224
        model_ft = model

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

def get_class_distribution(dataset_obj, idx2class):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    for _, label_id in dataset_obj:
        label = idx2class[label_id]
        count_dict[label] += 1
    return count_dict

def plot_from_dict(dict_obj, plot_title, **kwargs):
    return sns.barplot(data = pd.DataFrame.from_dict([dict_obj]).melt(), x = "variable", y="value", hue="variable", **kwargs).set_title(plot_title)

def get_class_distribution_loaders(dataloader_obj, dataset_obj, idx2class):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:    
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else: 
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1
    return count_dict

def multi_acc(y_pred, y_test):
    """ Function to calculate multi-class accuracy
    """
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc)
#    acc = torch.round(acc * 100)
    return acc

def make_train_step(model, model_name, loss_fn, optimizer):
    """ Function to make one training step
    """
    def perform_train_step(X_train_batch, y_train_batch):
        model.train()
        # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#inception-v3
        if model_name == "inception_v3":
            y_train_pred, aux_outputs = model(X_train_batch)
            y_train_pred = y_train_pred.squeeze()
            aux_outputs = aux_outputs.squeeze()
            train_loss_1 = loss_fn(y_train_pred, y_train_batch)
            train_loss_2 = loss_fn(aux_outputs, y_train_batch)
            train_loss = train_loss_1 + 0.4*train_loss_2
        else:
            y_train_pred = model(X_train_batch).squeeze()
            train_loss = loss_fn(y_train_pred, y_train_batch)

        train_acc = multi_acc(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return (train_acc.item(), train_loss.item())

    return perform_train_step

def make_val_step(model, model_name, loss_fn, optimizer):
    """ Function to make one validation step
    """
    def perform_val_step(X_val_batch, y_val_batch):
        model.eval()
        y_val_pred = model(X_val_batch).squeeze()
        val_acc = multi_acc(y_val_pred, y_val_batch)
        val_loss = loss_fn(y_val_pred, y_val_batch)
        return (val_acc.item(), val_loss.item())

    return perform_val_step

def mini_batch(device, data_loader, step_fn):
    """ Function to run through a mini-batch (train or validation)
    """
    mini_batch_acc_list = []
    mini_batch_loss_list = []
    for (X_batch, y_batch) in data_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        (mini_batch_acc, mini_batch_loss) = step_fn(X_batch, y_batch)

        mini_batch_acc_list.append(mini_batch_acc)
        mini_batch_loss_list.append(mini_batch_loss)

    loss = np.mean(mini_batch_loss_list)
    acc = np.mean(mini_batch_acc_list)
#    set_trace()
    return (acc, loss)

def get_image_transforms(input_size):
    image_transforms = {
        "train": transforms.Compose([
#            transforms.RandomResizedCrop(input_size),

#            transforms.Resize((input_size, input_size)),
#            transforms.RandomHorizontalFlip(),
#            transforms.RandomRotation(degrees=15),
#            transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
#            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),

            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15,),
            transforms.RandomAffine(degrees = 5, translate = (0.2, 0.2)),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
#            transforms.RandomPerspective(distortion_scale=0.3),
            transforms.RandomResizedCrop((input_size, input_size)),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
#            transforms.RandomErasing(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
#            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return image_transforms

def output_metadata_to_file(y_prob_arr, test_file_name_list, output_path, n_cls):
    with open(output_path, "wt") as fid:
        for file_idx, file_name in enumerate(test_file_name_list):
            fid.write("%s," % (file_name))
            for cls_idx in range(n_cls):
                if cls_idx < (n_cls-1):
                    fid.write("%.6f," % y_prob_arr[file_idx, cls_idx])
                else:
                    fid.write("%.6f" % y_prob_arr[file_idx, cls_idx])
            fid.write("\n")

def eval_model(test_loader, model):
    y_prob_list = []
    y_pred_list = []
    y_true_list = []
    test_file_name_list = []
    with torch.no_grad():
        # Only for batch_size=1 in test_loader
        for test_idx, (x_batch, y_batch) in enumerate(test_loader, 0):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_test_prob = model(x_batch)
            y_test_prob_cpu = y_test_prob.detach().cpu().numpy()
            y_test_prob_cpu = y_test_prob_cpu[0]
            y_test_prob_cpu = softmax(y_test_prob_cpu)
            y_pred_tag = np.argmax(y_test_prob_cpu)
            y_pred_list.append(y_pred_tag)

            y_true_list.append(y_batch.cpu().numpy())
            y_prob_list.append(y_test_prob_cpu)

            test_file_name = test_loader.dataset.samples[test_idx][0]
            test_file_name_list.append(test_file_name)
#            set_trace()
    y_true_list = [i[0] for i in y_true_list]
    y_prob_arr = np.array(y_prob_list)
    return (y_prob_arr, y_true_list, y_pred_list, test_file_name_list)

def run_pytorch_image_classification_model(
        root_dir, model_name, dataset_name, train_path, val_path, test_path, feature_extract,
        use_pretrained, n_epoch, is_train, result_folder, prefix_tag):
    np.random.seed(0)
    torch.manual_seed(0)
    print("We're using =>", DEVICE)
    print("The data lies here =>", root_dir)

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    model, input_size = initialize_model(
        model_name,
        len(os.listdir(pjoin(root_dir, "train"))),
        feature_extract, use_pretrained=True)
    model.to(DEVICE)
    print(model)
    model_output_name = '%s_best.pth' % (model_name)

    image_transforms = get_image_transforms(input_size)

    dataset_train = datasets.ImageFolder(
        root=train_path,
        transform=image_transforms["train"])
    dataset_val = datasets.ImageFolder(
        root=val_path,
        transform=image_transforms["val"])
    dataset_test = datasets.ImageFolder(
        root=test_path,
        transform = image_transforms["test"])

    cls_list = dataset_train.classes
    n_cls = len(cls_list)
    cls_to_idx_dict = dataset_train.class_to_idx
    idx2class = {v: k for k, v in dataset_train.class_to_idx.items()}

    train_loader = DataLoader(dataset=dataset_train, shuffle=True,
        batch_size=16)
    val_loader = DataLoader(dataset=dataset_val, shuffle=False,
        batch_size=4)
    test_loader = DataLoader(dataset=dataset_test,
        shuffle=False, batch_size=1)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
#    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
#    optimizer = optim.RMSprop(model.parameters(), lr=2e-5)
#    optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

    train_step_fn = make_train_step(model, model_name, criterion, optimizer)
    val_step_fn = make_val_step(model, model_name, criterion, optimizer)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100000

    if is_train is True:
        print("Begin training.")
#        for e in tqdm(range(1, n_epoch)):
        for e in range(1, n_epoch):
            # TRAINING
            (train_epoch_acc, train_epoch_loss) = mini_batch(DEVICE, train_loader, train_step_fn)
            loss_stats['train'].append(train_epoch_loss)
            accuracy_stats['train'].append(train_epoch_acc)

            with torch.no_grad():
                (val_epoch_acc, val_epoch_loss) = mini_batch(DEVICE, val_loader, val_step_fn)
                loss_stats['val'].append(val_epoch_loss)
                accuracy_stats['val'].append(val_epoch_acc)


            print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc:.3f}| Val Acc: {val_epoch_acc:.3f}')

#        if val_epoch_acc > best_acc:
        if val_epoch_loss < best_loss:
            best_acc = val_epoch_acc
            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

#        if (val_epoch_acc / len(val_loader)) > best_acc:
#            best_acc = val_epoch_acc / len(val_loader)
#            best_model_wts = copy.deepcopy(model.state_dict())

        # Plot loss and accuracy
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

        # Plot line charts
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
        sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
        plt.savefig(pjoin(result_folder, "%s_Loss_accuracy.png" % (model_name)))

        # Save model
        torch.save(best_model_wts, model_output_name)
    else:
        pass

    model.load_state_dict(torch.load(model_output_name))
    model.to(DEVICE)
    model.eval()

    if "Test" in prefix_tag:    
        pass
        # (val_y_prob_arr, val_y_true_list, val_y_pred_list, val_file_name_list) = eval_model(_loader)    
        # output_file_name = "metadata._%s_%s.txt" % (model_name, prefix_tag)
        # output_path = pjoin(result_folder, output_file_name)
        # output_metadata_to_file(y_prob_arr, test_file_name_list, output_path, n_cls)

    (y_prob_arr, y_true_list, y_pred_list, test_file_name_list) = eval_model(test_loader, model)    
    output_file_name = "metadata._%s_%s.txt" % (model_name, prefix_tag)
    output_path = pjoin(result_folder, output_file_name)
    output_metadata_to_file(y_prob_arr, test_file_name_list, output_path, n_cls)

    # Calculate metrics, such as accuracy, F1 score, etc.
    acc = accuracy_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list, average="macro")
    recall = recall_score(y_true_list, y_pred_list, average="macro")
    f1 = f1_score(y_true_list, y_pred_list, average="macro")
    conf = confusion_matrix(y_true_list, y_pred_list)

    print(classification_report(y_true_list, y_pred_list))

    np.savetxt(
        pjoin(result_folder, '%s_%s_conf.csv' % (dataset_name, model_name)),
        conf, fmt='%d', delimiter=',')
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,
                               display_labels=cls_list)
    disp.plot()
    plt.show()
    plt.savefig(
        pjoin(result_folder, '%s_%s_conf_visualize.png' % (dataset_name, model_name))
    )
    plt.close()

    with open(pjoin(result_folder, '%s_%s_result.csv' % (dataset_name, model_name)), 'wt') as fid:
        fid.write('Acc,F1\n')
        fid.write('%.5f,%.5f\n' % (acc, f1))

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    plt.savefig(pjoin(result_folder, "heatmap.png"))
    plt.close()
#    set_trace()

def main():
#    model_name = "resnet50"
#    model_list = ["vgg16", "resnet50", "resnet101", "inception_v3", "squeezenet", "densenet"]
#    model_list = ["vgg16", "resnet50", "resnet101", "inception_v3", "densenet"]
#    model_list = ["vgg16", "resnet50", "inception_v3", "densenet"]
#    model_list = ["vgg16", "resnet50", "inception_v3", "densenet"]
#    model_list = ["SwinTransformer"]
    model_list = ["VisionTransformer"]
#    model_list = ["MaxViT"]
#    model_list = ["Regnet"]
#    model_list = ["ConvNeXt"]
#    model_list = ["EfficientNetV2"]
#    model_list = ["ReXNet"]
#    model_list = ["NoisyStudent"]
#    model_list = ["inception_v3"]
#    model_list = ["inception_v3", "squeezenet", "densenet"]
    dataset_name = "Saibok_Total_balanced"
#    dataset_name = "FusionEnhanced+BackscatterRemoval_Saibok_2014_2017_balanced_train_val_test"
    root_dir = "/home/truong/Desktop/TRUONG/datasets/%s/" % (dataset_name)
    prefix_tag = "Test"
    feature_extract = False
    use_pretrained = True
    n_epoch = 200
    is_train = False

    for model_name in model_list:
        result_folder = "./Result_%s_%s" % (model_name, dataset_name)
        print("Running model: %s" % (model_name))
        train_path = pjoin(root_dir, "train")
        val_path = pjoin(root_dir, "val")
        test_path = pjoin(root_dir, "test")
        run_pytorch_image_classification_model(
            root_dir, model_name, dataset_name, train_path, val_path, test_path, feature_extract,
             use_pretrained, n_epoch, is_train, result_folder, prefix_tag)


if __name__ == "__main__":
    main()
