from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2 as cv
import json 

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "/data/feng/wc/wc_finetune/10"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for 
num_epochs = 50
# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # output_yet = False
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            true_negatives = 0.0
            true_positives = 0.0
            false_negatives = 0.0
            false_positives = 0.0
            output_idx = 0
            all_preds = []
            all_labels = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                TP = 0 # label 0, predict 0
                FN = 0 # label 0, predict 1
                TN = 0 # label 1, predict 1
                FP = 0 # label 1, predict 0
                confusion_vector = preds.cpu().numpy() / labels.data.cpu().numpy()
                # Element-wise division of the 2 tensors returns a new tensor which holds a
                # unique value for each case:
                #   1     where prediction and truth are 1 (True Negative)
                #   inf   where prediction is 1 and truth is 0 (False Negative)
                #   nan   where prediction and truth are 0 (True Positive)
                #   0     where prediction is 0 and truth is 1 (False Positive)
                # print(preds)
                # print(labels.data)
                # print(confusion_vector)
                true_negatives += np.sum(confusion_vector == 1)
                false_negatives += np.sum(confusion_vector == float('inf'))
                true_positives += np.sum(np.isnan(confusion_vector))
                false_positives += np.sum(confusion_vector == 0)

                
                # store examples
                tn = confusion_vector == 1
                fn = confusion_vector == float('inf')
                tp = np.isnan(confusion_vector)
                fp = confusion_vector == 0
                # print(labels)
                for i in range(len(tn)):
                    if tn[i]:
                        output_dir = './res_correct/'
                    elif fn[i]:
                        output_dir = './nonres_wrong/'
                    elif tp[i]:
                        output_dir = './nonres_correct/'
                    else:
                        output_dir = './res_wrong/'
                    image = inputs[i].cpu().numpy()
                    image = np.moveaxis(image, 0, -1)
                    image = image - np.min(image)
                    image = image / np.max(image)
                    image = np.uint8(255 * image)
                    cv.imwrite(output_dir + str(output_idx) + '.jpg', image)
                    output_idx += 1
                # output_yet = True
                # store all predicted accuracies 
                outputs = torch.nn.functional.softmax(outputs)
                all_preds.extend(outputs.cpu().numpy().tolist())
                all_labels.extend(labels.data.cpu().numpy().tolist())
            if not epoch == 0:
                continue
            # finish iterating through all data
            with open('preds_0.json', 'w') as f:
                json.dump(all_preds, f)
            with open('labels_0.json', 'w') as f:
                json.dump(all_labels, f)
            


            
            TPR = true_positives / (true_positives + false_negatives)
            TNR = true_negatives / (true_negatives + false_positives)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} TPR: {:.4f} TNR: {:.4f}'.format(phase, TPR, TNR))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
                # torch.save(model.state_dict(), "%s/10_best.weights" % ('checkpoints'))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224
        # Load Pre-trained model on fmow dataset 
        model_ft.load_state_dict(torch.load('checkpoints/best_fmow.weights'))

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
