from .resnet import resnet50


def build_backbone(**kwargs):
    return resnet50()
