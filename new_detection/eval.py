import torch
from utils import *
from model import fasterrcnn_resnet18_fpn
from torch.utils.data import DataLoader
from dataset import BuildingDetectionDataset, custom_collate_fn
import torchvision.transforms as transforms
import json
import pdb

# Output structure:
# building_id: unique
# region: Henderson
# label: residential/non-residential
# prediction region: loop over all prediction to calculate this
# label region: 
# IoU: intersection over union
# residential likelihood: P
# non-residential likelihood: 1 - P

def find_best_box(target, preds, iou_threshold):
    best_pred = -1
    for pred_idx, pred in enumerate(preds):
        iou = calculate_iou(target, pred).item()
        if iou > iou_threshold:
            best_pred = pred_idx
    return best_pred

def calculate_iou(target, pred):
    # target/pred: tensor[x1, y1, x2, y2]
    # 1 x 4
    target_tensor = target.unsqueeze(0)
    pred_tensor = pred.unsqueeze(0)
    iou = find_jaccard_overlap(target_tensor, pred_tensor)
    return iou

data_dir = '/data/feng/building-detect/'
pretrained_weight = 'checkpoints/12_14.pth.tar'
device = torch.device('cuda:2')
batch_size = 10
eval_size = 5000
iou_threshold = 0.2

model = fasterrcnn_resnet18_fpn(num_classes=2, pretrained_backbone=True)
model.load_state_dict(torch.load(pretrained_weight))
model.to(device)
model.eval()

val_loader = DataLoader(BuildingDetectionDataset(
    '/home/yuansong/code/building/new_detection/config/test_config.txt',
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])),batch_size=batch_size, shuffle=False, pin_memory=False, 
    collate_fn=custom_collate_fn 
)
results = []
building_id = 0
for batch_idx, (inputs, targets) in enumerate(val_loader):
    # [N x (C x W x H)]
    inputs = [input_.to(device) for input_ in inputs]
    # [N x {boxes, labels, area, iscrowd}]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    outputs = model(inputs)    
    for target_idx, target in enumerate(targets):
        for building_idx, gt_box in enumerate(target['boxes']):
            pred_box_idx = find_best_box(gt_box, outputs[target_idx]['boxes'].detach(), iou_threshold)
            if pred_box_idx == -1:
                pred_box = torch.FloatTensor([])
                iou = 0
                res_prob = -1
                non_res_prob = -1
            else:
                pred_box = outputs[target_idx]['boxes'][pred_box_idx].detach()
                iou = calculate_iou(gt_box, pred_box).item()
                res_prob = outputs[target_idx]['scores'][pred_box_idx].item()
                non_res_prob = 1 - res_prob
            label = target['labels'][building_idx].item()
            building_result = {
                "building_id": building_id,
                "region": "Henderson",
                "groundtruth label": label,
                "prediction box": pred_box.tolist(),
                "groundtruth box": gt_box.detach().tolist(),
                "IoU": iou,
                "residential likelihood": res_prob,
                "non-residential likelihood": non_res_prob,
            }
            results.append(building_result)
            building_id += 1
            print(building_id)
    if building_id >= eval_size:
        break
with open('eval_result.json', 'w') as f:
    json.dump(results, f)
    print('evaluation results are dumped to eval_result.json')