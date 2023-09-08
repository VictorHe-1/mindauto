from .decoder import DetectionTransformerDecoder
from .encoder import BEVFormerEncoder, BEVFormerLayer
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .transformer import FFN

transformer_layers = {
    'BEVFormerLayer': BEVFormerLayer,
    'MyCustomBaseTransformerLayer': MyCustomBaseTransformerLayer
}
transformer_layers_sequence = {
    'DetectionTransformerDecoder': DetectionTransformerDecoder,
    'BEVFormerEncoder': BEVFormerEncoder
}
attention_layers = {
    'TemporalSelfAttention': TemporalSelfAttention,
    'MSDeformableAttention3D': MSDeformableAttention3D
}
feedforward_layers = {
    'FFN': FFN
}


def build_transformer_layer(cfg):
    obj_cls = transformer_layers.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def build_transformer_layer_sequence(cfg):
    obj_cls = transformer_layers_sequence.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def build_attention(cfg):
    obj_cls = attention_layers.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)


def build_feedforward_network(cfg):
    obj_cls = feedforward_layers.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
