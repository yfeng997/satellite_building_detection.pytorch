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
from dataset import BuildingDetectionDataset, custom_collate_fn
from utils import *

import pdb
    
def train_or_eval(model, dataloader, optimizer=None, train=True, device=torch.device('cuda:0')):
    avg_meter = meter.AverageValueMeter()
    if train:
        model.train()
    else:
        model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # [N x (C x W x H)]
        inputs = [input_.to(device) for input_ in inputs]
        # [N x {boxes, labels, area, iscrowd}]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # model behaves differently during training vs. evaluation
        if train:
            loss_dict = model(inputs, targets)
            loss = sum(los for los in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_meter.add(loss.item())
        else:
            outputs = model(inputs)  
            mAP = calculate_metric(outputs, targets, device)
            avg_meter.add(mAP)

        if batch_idx % 100 == 0:
            meter_name = 'total loss' if train else 'mAP'
            print('batch: %i %s: %f' % (batch_idx, meter_name, avg_meter.mean))
    return avg_meter.mean

# predictions and targets are both torch tensors 
def calculate_metric(predictions, targets, device):
    '''
    predictions/targets: [{
        boxes: [[x1, y1, x2, y2], ...], 
        labels: [0/1, ...],
        scores: [0.53, 0.23, ...]
    }, ...]
    '''
    average_precisions, mean_average_precision = calculate_mAP(
        det_boxes=[p['boxes'] for p in predictions], 
        det_labels=[p['labels'] for p in predictions], 
        det_scores=[p['scores'] for p in predictions], 
        true_boxes=[t['boxes'] for t in targets], 
        true_labels=[t['labels'] for t in targets], 
        true_difficulties=[torch.IntTensor([0 for l in t['labels']]) for t in targets],
        device=device
    )
    return mean_average_precision
    
'''
TODO: check if PyTorch has built-in logging/visualization modules 
'''
# def output_history_graph(train_acc_history, val_acc_history, train_loss_history, val_loss_history):
#     epochs = len(train_acc_history)
#     plt.figure(0)
#     plt.plot(list(range(epochs)), train_acc_history, label='train')
#     plt.plot(list(range(epochs)), val_acc_history, label='val')
#     plt.legend(loc='upper left')
#     plt.savefig('acc.png')
#     plt.clf()

#     plt.figure(1)
#     plt.plot(list(range(epochs)), train_loss_history, label='train')
#     plt.plot(list(range(epochs)), val_loss_history, label='val')
#     plt.legend(loc='upper left')
#     plt.savefig('loss.png')
#     plt.clf()

# parameters
data_dir = '/data/feng/building-detect/'
checkpoint_dir = 'checkpoints'
learning_rate = 1e-4
weight_decay = 1e-4
batch_size = 10
num_epochs = 50
# pretrained_weight = 'checkpoints/best.pth.tar'
device = torch.device('cuda:0')

# load model 
model = models.detection.fasterrcnn_resnet50_fpn(num_classes=2, pretrained_backbone=True)
# if pretrained_weight:
#     model.load_state_dict(torch.load(pretrained_weight))
model.to(device)

# define data transform and data loader 
train_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/train_config.txt',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])), batch_size=batch_size, shuffle=True, pin_memory=True, 
    collate_fn=custom_collate_fn 
)
val_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/test_config.txt',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])),batch_size=batch_size, shuffle=True, pin_memory=True, 
    collate_fn=custom_collate_fn 
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# training loop 
train_acc_hist = []
train_loss_hist = []
val_acc_hist = []
val_loss_hist = []
best_acc = 0.0
for epoch in range(num_epochs):
    loss = train_or_eval(model, train_loader, optimizer, train=True, device=device)
    # train_acc_hist.append(acc)
    # train_loss_hist.append(loss)
    acc = train_or_eval(model, val_loader, train=False, device=device)
    # val_acc_hist.append(acc)
    # val_loss_hist.append(loss)

    if acc > best_acc:
        save_path = os.path.join(checkpoint_dir, 'best_acc.pth.tar')
        torch.save(model.state_dict(), save_path)
        best_acc = acc
        print('model with accuracy %f saved to path %s' % (acc, save_path))
    
    print('****** epoch: %i val loss: %f val acc: %f best_acc: %f ******' % (epoch, loss, acc, best_acc))

    # output_history_graph(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)
