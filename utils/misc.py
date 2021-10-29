import torch
import utils.models.backbones
from utils.models.deeplabV3 import DeepLabV3
from yacs.config import CfgNode
from utils.constants import IMAGE_MEAN, IMAGE_STDDEV, HEX_COLORS
from torchvision.utils import draw_segmentation_masks


def load_config(fname):
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(fname)
    cfg.freeze()
    return cfg


def build_model(cfg):
    backbone = getattr(utils.models.backbones, cfg.backbone.pop('name'))(**cfg.backbone)
    return DeepLabV3(
        backbone=backbone,
        input_size=cfg.input_size,
        num_classes=cfg.num_classes,
        aspp_rates=cfg.aspp_rates,
        image_pooling=cfg.image_pooling,
        dropout_rate=cfg.dropout_rate
    )


def unnormalize(images):
    mean = torch.FloatTensor(IMAGE_MEAN).reshape([3, 1, 1])
    stddev = torch.FloatTensor(IMAGE_STDDEV).reshape([3, 1, 1])
    images = torch.clip(images * stddev + mean, 0, 255)
    images = images.byte()
    return images


def draw(image, labels, num_classes, alpha=0.8):
    """
    Args:
        image: uint8 tensor. Shape: [C, H, W].
        labels: int64 tensor. Shape: [H, W].
    """
    classes = torch.arange(num_classes).reshape([-1, 1, 1])
    masks = (labels == classes)   # [num_classes, H, W]
    colors = HEX_COLORS[:num_classes]
    return draw_segmentation_masks(image, masks, alpha=alpha, colors=colors)
