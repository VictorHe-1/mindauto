import os
import sys
import time
import os.path as osp
import sklearn  # avoid memory allocation bugs
from addict import Dict
import logging

import mindspore as ms
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from tools.arg_parser import parse_args_and_config
args, config = parse_args_and_config()

from mindauto.utils.logger import set_logger
from mindauto.utils.seed import set_seed
from mindauto.data import build_dataset
from mindauto.models import build_model

logger = logging.getLogger("mindauto.test")


def main(cfg):
    ms.set_context(mode=cfg.system.mode, device_id=0, device_target='Ascend', pynative_synchronize=True)
    device_num = None
    rank_id = None

    # create logger, only rank0 log will be output to the screen
    set_logger(
        name="mindauto",
        output_dir=cfg.train.ckpt_save_dir,
        rank=0,
        log_level=eval(cfg.system.get("log_level", "logging.INFO")),
    )
    if "DEVICE_ID" in os.environ:
        logger.info(
            f"Standalone testing. Device id: {os.environ.get('DEVICE_ID')}, "
            f"specified by environment variable 'DEVICE_ID'."
        )
    else:
        device_id = cfg.system.get("device_id", 0)
        ms.set_context(device_id=device_id)
        logger.info(
            f"Standalone testing. Device id: {device_id}, "
            f"specified by system.device_id in yaml config file or is default value 0."
        )

    set_seed(cfg.system.seed)
    # create dataset
    dataset, loader = build_dataset(
        cfg.eval.dataset,
        cfg.eval.loader,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=False,
        refine_batch_size=True,
    )
    amp_level = cfg.system.get("amp_level", "O0")
    network = build_model(cfg.model, ckpt_load_path=cfg.eval.ckpt_load_path,
                          amp_level=amp_level)
    network.set_train(False)
    network.img_backbone.train(False)

    outputs = []
    num_batches = loader.get_dataset_size()
    data_iterator = loader.create_tuple_iterator(output_numpy=False, do_copy=False)
    for in_data in tqdm(data_iterator, total=num_batches):
        output = network(*in_data)
        outputs.append(output[0])
    kwargs = {}
    kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
        '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
    eval_configs = cfg.get('eval', {}).copy()
    # hard-code way to remove EvalHook args
    eval_kwargs = {}
    eval_kwargs['pipeline'] = eval_configs['dataset']['pipeline']
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    config = Dict(config)
    main(config)
