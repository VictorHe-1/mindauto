from .match_cost import FocalLossCost, IoUCost, BBox3DL1Cost

match_cost_type = {
    'IoUCost': IoUCost, 'FocalLossCost': FocalLossCost,
    'BBox3DL1Cost': BBox3DL1Cost}


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    obj_cls = match_cost_type.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
