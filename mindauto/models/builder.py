import mindspore as ms
from mindspore.amp import auto_mixed_precision

from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck
from .bevformer import BEVFormer

model_types = {
    'BEVFormer': BEVFormer
}


def build_model(cfg, **kwargs):
    obj_cls = model_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    model = obj_cls(**args)
    if kwargs['ckpt_load_path'] is not None:
        ms.load_checkpoint(kwargs['ckpt_load_path'], model)

    if 'amp_level' in kwargs:
        auto_mixed_precision(model, amp_level=kwargs["amp_level"])
    return model
