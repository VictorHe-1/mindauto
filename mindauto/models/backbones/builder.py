import mindspore as ms

from .resnet import ResNet

backbone_types = {'ResNet': ResNet}


def build_backbone(**kwargs):
    obj_cls = backbone_types.get(kwargs['type'])
    args = kwargs.copy()
    args.pop('type')
    load_path = args.pop('ckpt_load_path', None)
    if load_path is None:
        return obj_cls(**args)
    else:
        model = obj_cls(**args)
        ms.load_checkpoint(kwargs['ckpt_load_path'], model)
        return model
