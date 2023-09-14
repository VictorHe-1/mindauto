from .decoder import DetectionTransformerDecoder, CustomMSDeformableAttention
from .encoder import BEVFormerEncoder, BEVFormerLayer
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer, DetrTransformerDecoderLayer
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D, SpatialCrossAttention
from .transformer import FFN, PerceptionTransformer, MultiheadAttention

transformer_layers = {
    'BEVFormerLayer': BEVFormerLayer,
    'MyCustomBaseTransformerLayer': MyCustomBaseTransformerLayer,
    'DetrTransformerDecoderLayer': DetrTransformerDecoderLayer
}
transformer_layers_sequence = {
    'DetectionTransformerDecoder': DetectionTransformerDecoder,
    'BEVFormerEncoder': BEVFormerEncoder
}
attention_layers = {
    'TemporalSelfAttention': TemporalSelfAttention,
    'MSDeformableAttention3D': MSDeformableAttention3D,
    'SpatialCrossAttention': SpatialCrossAttention,
    'MultiheadAttention': MultiheadAttention,
    'CustomMSDeformableAttention': CustomMSDeformableAttention
}
feedforward_layers = {
    'FFN': FFN
}
transformer_types = {
    'PerceptionTransformer': PerceptionTransformer
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


def build_transformer(cfg):
    obj_cls = transformer_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
