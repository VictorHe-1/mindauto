from .resnet import ResNet

backbone_types = {'ResNet': ResNet}


def build_backbone(**kwargs):
    obj_cls = backbone_types.get(kwargs['type'])
    args = kwargs.copy()
    args.pop('type')
    return obj_cls(**args)
