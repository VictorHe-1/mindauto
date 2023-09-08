from .nms_free_coder import NMSFreeCoder

bbox_coder_types = {
    'NMSFreeCoder': NMSFreeCoder,
}


def build_bbox_coder(cfg):
    obj_cls = bbox_coder_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
