from .positional_encoding import LearnedPositionalEncoding
from .bevformer_head import BEVFormerHead
positional_encodings = {
    'LearnedPositionalEncoding': LearnedPositionalEncoding
}
head_types = {
    'BEVFormerHead': BEVFormerHead
}


def build_positional_encoding(cfg):
    """Builder for Position Encoding."""
    obj_cls = positional_encodings.get(cfg[type])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def build_head(cfg):
    """Builder for Position Encoding."""
    obj_cls = head_types.get(cfg[type])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
