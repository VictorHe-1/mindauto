from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck
from .bevformer import BEVFormer

model_types = {
    'BEVFormer': BEVFormer
}


def build_model(cfg):
    obj_cls = model_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
