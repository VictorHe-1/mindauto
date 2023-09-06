from mindspore import nn

activation_layers = {
    'ReLU': nn.ReLU, 'LeakyReLU': nn.LeakyReLU,
    'Sigmoid': nn.Sigmoid, 'Tanh': nn.Tanh, 'ELU': nn.ELU,
    'PReLU': nn.PReLU, 'RReLU': nn.RReLU}


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    obj_cls = activation_layers.get(cfg[type])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
