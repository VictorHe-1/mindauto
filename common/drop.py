from mindspore import nn

dropout_layers = {
    'Dropout': nn.Dropout
}


def build_dropout(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    obj_cls = dropout_layers.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    return obj_cls(**args)
