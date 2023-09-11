from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ['BEVFormer', 'bevformer_tiny']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 640, 640),
        **kwargs
    }


default_cfgs = {
    'bevformer_tiny': _cfg(
        url='https://download.mindspore.cn/toolkits/mindocr/east/east_resnet50_ic15-7262e359.ckpt'),  # TODO: change to BEVFormer
}


class BEVFormer(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def bevformer_tiny(pretrained=False, **kwargs):
    pretrained_backbone = not pretrained
    model_config = {
        "backbone": {
            'name': 'det_resnet50',
            'pretrained': pretrained_backbone
        },
        "neck": {
            "name": 'EASTFPN',
            "out_channels": 128
        },
        "head": {
            'name': 'EASTHead'
        }
    }
    model = BEVFormer(model_config)

    # load pretrained weights
    if pretrained:
        default_cfg = default_cfgs['bevformer_tiny']
        load_pretrained(model, default_cfg)

    return model
