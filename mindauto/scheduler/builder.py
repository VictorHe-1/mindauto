"""Scheduler Factory"""
import logging

from .dynamic_lr import (
    cosine_annealing_lr,
)

__all__ = ["create_scheduler"]
_logger = logging.getLogger(__name__)


def create_scheduler(
    steps_per_epoch: int,
    scheduler: str = "CosineAnnealing",
    lr: float = 0.01,
    min_lr_ratio: float = 1e-3,
    warmup: str = 'linear',
    warmup_iters: int = 3,
    warmup_ratio: float = 0.33,
    num_epochs: int = 200,
):
    if isinstance(lr, str):
        lr = eval(lr)
    if isinstance(min_lr_ratio, str):
        min_lr_ratio = eval(min_lr_ratio)
    if isinstance(warmup_ratio, str):
        warmup_ratio = eval(warmup_ratio)
    if scheduler == "CosineAnnealing":
        main_lr_scheduler = cosine_annealing_lr(
            lr=lr,
            warmup=warmup,
            warmup_iters=warmup_iters,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs
        )
    else:
        raise NotImplementedError
    # combine
    lr_scheduler = main_lr_scheduler

    return lr_scheduler
