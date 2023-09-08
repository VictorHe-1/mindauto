from .hungarian_assigner_3d import HungarianAssigner3D

assigner_types = {
    'HungarianAssigner3D': HungarianAssigner3D
}


def build_assigner(cfg):
    obj_cls = assigner_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
