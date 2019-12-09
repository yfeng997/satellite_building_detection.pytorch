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
import cv2
from dataset import BuildingDetectionDataset, custom_collate_fn
from utils import *
from model import fasterrcnn_resnet18_fpn
from datetime import datetime
import time
import pdb
    
def get_loss(model, inputs, targets, optimizer, backward=True):
    model.train()
    loss_dict = model(inputs, targets)
    loss = sum(los for los in loss_dict.values())
    if backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def get_acc(model, inputs, targets):
    model.eval()
    outputs = model(inputs)
    cpu_device = torch.device('cpu')
    mAP = calculate_metric(outputs, targets, device=cpu_device)
    return mAP, outputs

def train_or_eval(model, dataloader, optimizer=None, train=True, device=torch.device('cuda:0'), store=False):
    acc_meter = meter.AverageValueMeter()
    loss_meter = meter.AverageValueMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        start_time = time.time()
        # [N x (C x W x H)]
        inputs = [input_.to(device) for input_ in inputs]
        # [N x {boxes, labels, area, iscrowd}]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        mAP, preds = get_acc(model, inputs, targets)
        acc_meter.add(mAP)
        loss = get_loss(model, inputs, targets, optimizer, backward=train)
        loss_meter.add(loss)
        
        if (not train) and store:
            store_results(inputs, preds, targets)
        if batch_idx % 10 == 0:
            tag = 'train' if train else 'val'
            curr_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            delta_time = (time.time() - start_time) * 48
            print('%s %s batch: %i loss: %f acc: %f epoch time: %f' % (curr_time, tag, batch_idx, loss_meter.mean, acc_meter.mean, delta_time))
        if (not train) and batch_idx == 20:
            return loss_meter.mean, acc_meter.mean
        if batch_idx == 100:
            return loss_meter.mean, acc_meter.mean
    return loss_meter.mean, acc_meter.mean

# predictions and targets are both torch tensors 
def calculate_metric(predictions, targets, device=torch.device('cpu')):
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
    
def output_history_graph(train_hist, val_hist):
    epochs = len(train_hist)
    plt.figure(0)
    plt.plot(list(range(epochs)), [x[0] for x in train_hist], label='train')
    plt.plot(list(range(epochs)), [x[0] for x in val_hist], label='val')
    plt.legend(loc='upper left')
    plt.savefig('loss.jpg')
    plt.clf()

    plt.figure(1)
    plt.plot(list(range(epochs)), [x[1] for x in train_hist], label='train')
    plt.plot(list(range(epochs)), [x[1] for x in val_hist], label='val')
    plt.legend(loc='upper left')
    plt.savefig('acc.jpg')
    plt.clf()

# parameters
data_dir = '/data/feng/building-detect/'
checkpoint_dir = 'checkpoints'
lr = 5e-5
weight_decay = 0
batch_size = 16
num_epochs = 1000
pretrained_weight = 'checkpoints/best_acc.pth.tar'
device = torch.device('cuda:1')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# load model 
model = fasterrcnn_resnet18_fpn(num_classes=2, pretrained_backbone=True)
model.load_state_dict(torch.load(pretrained_weight))
model.to(device)
# define data transform and data loader 
train_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/train_config.txt',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])), batch_size=batch_size, shuffle=True, pin_memory=False, 
    collate_fn=custom_collate_fn 
)
val_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/test_config.txt',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])),batch_size=batch_size, shuffle=True, pin_memory=False, 
    collate_fn=custom_collate_fn 
)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training loop 
train_hist = []
val_hist = []
best_acc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_or_eval(model, train_loader, optimizer, train=True, device=device)
    train_hist.append([train_loss, train_acc])
    val_loss, val_acc = train_or_eval(model, val_loader, train=False, device=device, store=False)
    val_hist.append([val_loss, val_acc])

    if val_acc > best_acc:
        save_path = os.path.join(checkpoint_dir, 'best_acc.pth.tar')
        torch.save(model.state_dict(), save_path)
        best_acc = val_acc
        print('model with val accuracy %f saved to path %s' % (val_acc, save_path))
    
    print('****** epoch: %i train loss: %f train acc: %f val loss: %f val acc: %f best_acc: %f ******' % (epoch, train_loss, train_acc, val_loss, val_acc, best_acc))
    output_history_graph(train_hist, val_hist)
