from .focal_loss import FocalLoss
from .iou_loss import GIoULoss
from .smooth_l1_loss import L1Loss

loss_types = {
    'FocalLoss': FocalLoss,
    'GIoULoss': GIoULoss,
    'L1Loss': L1Loss
}


def build_loss(cfg, **kwargs):
    obj_cls = loss_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
