from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=250, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="/data/feng/building-detect/images", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/res.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/res.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=30, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path, img_size=opt.img_size)
model.apply(weights_init_normal)
model.load_state_dict(torch.load(opt.weights_path))
# load [Epoch 0/150, Batch 0/242] [Losses: x 0.263783, y 0.242547, w 7.087229, h 5.045750, conf 4.708252, cls 0.039098, total 17.386658, recall: 0.07182, precision: 0.00829]
# without [Epoch 0/150, Batch 0/242] [Losses: x 0.291999, y 0.240591, w 3.382115, h 3.164990, conf 4.359457, cls 0.063323, total 11.502476, recall: 0.12891, precision: 0.00100]

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path, img_size=opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# catch up to trained model's optimizer step
for i in range(120 * 242):
    optimizer.zero_grad()
    optimizer.step()

for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        if (batch_i % 100) == 0:
            print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch,
                    opt.epochs,
                    batch_i,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), "%s/%d.weights" % (opt.checkpoint_dir, epoch))
        # print('weight is saved as %d.weights' % epoch)
        # model.load_state_dict(torch.load(opt.weights_path))
        # loss = model(imgs, targets)
        # print('printing saved weight loss: %f' % loss.item())
