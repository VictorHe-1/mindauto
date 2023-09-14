from .fpn import FPN

neck_types = {
    'FPN': FPN,
}


def build_neck(cfg, **kwargs):
    obj_cls = neck_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
