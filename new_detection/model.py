from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn as nn
import pdb

def fasterrcnn_resnet18_fpn(num_classes=2, pretrained_backbone=True, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-18-FPN backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    Example::
        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    backbone = resnet_fpn_backbone('resnet18', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    # Modifications make the model smaller -- lessen overfitting
    # model.backbone.body.layer3 = nn.Sequential()
    # model.backbone.body.layer4 = nn.Sequential()
    # model.backbone.fpn.inner_blocks[1] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
    # model.backbone.fpn.inner_blocks[2] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
    # model.backbone.fpn.inner_blocks[3] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
    # model.backbone.fpn.layer_blocks[0] = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model.backbone.fpn.layer_blocks[1] = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model.backbone.fpn.layer_blocks[2] = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model.backbone.fpn.layer_blocks[3] = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model.rpn.head.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model.rpn.head.cls_logits = nn.Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1))
    # model.rpn.head.bbox_pred = nn.Conv2d(128, 12, kernel_size=(1, 1), stride=(1, 1))
    # model.rpn.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # model.rpn.cls_logits = nn.Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1))
    # model.rpn.bbox_pred = nn.Conv2d(128, 12, kernel_size=(1, 1), stride=(1, 1))
    # model.roi_heads.box_head.fc6 = nn.Linear(in_features=6272, out_features=256, bias=True)
    # model.roi_heads.box_head.fc7 = nn.Linear(in_features=256, out_features=256, bias=True)
    # model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=256, out_features=2, bias=True)
    # model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=256, out_features=8, bias=True)
    return model