from mindspore import nn

activation_layers = {
    'LN': nn.LayerNorm
}


def build_norm_layer(cfg):
    obj_cls = activation_layers.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
