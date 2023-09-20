import argparse

import yaml


def create_parser():
    parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        required=True,
        help="YAML config file specifying default arguments (default=" ")",
    )
    parser.add_argument(
        "-o",
        "--opt",
        nargs="+",
        help="Options to change yaml configuration values, "
        "e.g. `-o system.distribute=False eval.dataset.dataset_root=/my_path/to/ocr_data`",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )

    return parser


def _parse_options(opts: list):
    """
    Args:
        opt: list of str, each str in form f"{key}={value}"
    """
    options = {}
    if not opts:
        return options
    for opt_str in opts:
        assert (
            "=" in opt_str
        ), "Invalid option {}. A valid option must be in the format of {{key_name}}={{value}}".format(opt_str)
        k, v = opt_str.strip().split("=")
        options[k] = yaml.load(v, Loader=yaml.Loader)

    return options


def _merge_options(config, options):
    """
    Merge options (from CLI) to yaml config.
    """
    for opt in options:
        value = options[opt]

        # parse hierarchical key in option, e.g. eval.dataset.dataset_root
        hier_keys = opt.split(".")
        assert hier_keys[0] in config, f"Invalid option {opt}. The key {hier_keys[0]} is not in config."
        cur = config[hier_keys[0]]
        for level, key in enumerate(hier_keys[1:]):
            if level == len(hier_keys) - 2:
                assert key in cur, f"Invalid option {opt}. The key {key} is not in config."
                cur[key] = value
            else:
                assert key in cur, f"Invalid option {opt}. The key {key} is not in config."
                cur = cur[key]  # go to next level

    return config


def parse_args_and_config():
    """
    Return:
        args: command line argments
        cfg: train/eval config dict
    """
    parser = create_parser()
    args = parser.parse_args()  # CLI args

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # TODO: check validity of config arguments to avoid invalid config caused by typo.
        # _check_cfgs_in_parser(cfg, parser)
        # parser.set_defaults(**cfg)
        # parser.set_defaults(config=args_config.config)

    if args.opt:
        options = _parse_options(args.opt)
        cfg = _merge_options(cfg, options)

    return args, cfg
