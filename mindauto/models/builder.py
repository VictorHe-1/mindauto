from mindspore.amp import auto_mixed_precision

from .backbones import build_backbone
from .heads import build_head
from .necks import build_neck
from .bevformer import BEVFormer

model_types = {
    'BEVFormer': BEVFormer
}


def build_model(cfg, **kwargs):
    obj_cls = model_types.get(cfg['type'])
    args = cfg.copy()
    args.pop('type')
    model = obj_cls(**args)
    if "ckpt_load_path" in kwargs:
        load_from = kwargs["ckpt_load_path"]
        if isinstance(load_from, bool):
            raise ValueError(
                "Cannot find the pretrained checkpoint for a customized model without giving the url or local path "
                "to the checkpoint.\nPlease specify the url or local path by setting `model-pretrained` (if training) "
                "or `eval-ckpt_load_path` (if evaluation) in the yaml config"
            )

        # load_model(model, load_from) TODO: load_model

    if 'amp_level' in kwargs:
        auto_mixed_precision(model, amp_level=kwargs["amp_level"])
    return model
