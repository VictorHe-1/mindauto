from .random_sampler import RandomSampler
from .pseudo_sampler import PseudoSampler

sampler_types = {
    'RandomSampler': RandomSampler,
    'PseudoSampler': PseudoSampler
}


def build_sampler(cfg):
    obj_cls = sampler_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
