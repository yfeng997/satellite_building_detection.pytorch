# Training script to detect and classify residential buildings
# on satellite images. 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torchnet.meter as meter
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import cv2
from dataset import BuildingDetectionDataset

import pdb
    
def train_or_eval(model, dataloader, criterion, optimizer=None, train=True, device=torch.device('cuda:0')):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()
    if train:
        model.train()
    else:
        model.eval()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # [N x (C x W x H)]
        inputs = [input.to(device) for input in inputs]
        # [N x {boxes, labels, area, iscrowd}]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # model behaves differentlt during training vs. evaluation
        if train:
            loss_dict = model(inputs, targets)
            loss = sum(los for los in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            outputs = model(inputs)            
        
        acc = calculate_metric(outputs, targets)
        acc_meter.add(acc)
        loss_meter.add(loss.item())

        if batch_idx % 100 == 0:
            print('batch: %i loss: %f acc: %f' % (batch_idx, loss_meter.mean, acc_meter.mean))
    return loss_meter.mean, acc_meter.mean

# predictions and targets are both torch tensors 
def calculate_acc(predictions, targets):
    return (predictions == targets).sum().item() / predictions.size(0)
    
'''
TODO: check if PyTorch has built-in logging/visualization modules 
'''
def output_history_graph(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
    epochs = len(train_acc_history)
    plt.figure(0)
    plt.plot(list(range(epochs)), train_acc_history, label='train')
    plt.plot(list(range(epochs)), val_acc_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('acc.png')
    plt.clf()

    plt.figure(1)
    plt.plot(list(range(epochs)), train_loss_history, label='train')
    plt.plot(list(range(epochs)), val_loss_history, label='val')
    plt.legend(loc='upper left')
    plt.savefig('loss.png')
    plt.clf()

# parameters
data_dir = '/data/feng/building-detect/'
checkpoint_dir = 'checkpoints'
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 64
num_epochs = 150
# pretrained_weight = 'unittests/resnet/checkpoints/acc_82.pth.tar'
device = torch.device('cuda:0')

# load model 
model = models.detection.fasterrcnn_resnet50_fpn(num_classes=2)
if pretrained_weight is not None:
    model.load_state_dict(torch.load(pretrained_weight))
model.to(device)

# define data transform and data loader 
train_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/train_config.txt'
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4 
)
val_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/test_config.txt'
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])),batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# training loop 
train_acc_hist = []
train_loss_hist = []
val_acc_hist = []
val_loss_hist = []
best_acc = 0.0
for epoch in range(num_epochs):
    loss, acc = train(model, train_loader, criterion, optimizer, device=device)
    train_acc_hist.append(acc)
    train_loss_hist.append(loss)

    loss, acc = evaluate(model, val_loader, criterion, train=False, device=device)
    val_acc_hist.append(acc)
    val_loss_hist.append(loss)

    if acc > best_acc:
        save_path = os.path.join(checkpoint_dir, 'best_acc.pth.tar')
        # torch.save(model.state_dict(), save_path)
        best_acc = acc
        print('model with accuracy %f saved to path %s' % (acc, save_path))
    
    print('****** epoch: %i val loss: %f val acc: %f best_acc: %f ******' % (epoch, loss, acc, best_acc))

    output_history_graph(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)
