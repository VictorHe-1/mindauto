from math import cos, pi


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def get_warmup_lr(base_lr, cur_iters, warmup, warmup_ratio, warmup_iters):
    if warmup == 'constant':
        warmup_lr = base_lr * warmup_ratio
    elif warmup == 'linear':
        k = (1 - cur_iters / warmup_iters) * (1 -
                                                warmup_ratio)
        warmup_lr = base_lr * (1 - k)
    elif warmup == 'exp':
        k = warmup_ratio ** (1 - cur_iters / warmup_iters)
        warmup_lr = base_lr * k
    return warmup_lr


def cosine_annealing_lr(lr,
                        warmup,
                        warmup_iters,
                        warmup_ratio,
                        min_lr_ratio,
                        steps_per_epoch,
                        epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    target_lr = lr * min_lr_ratio
    for i in range(steps):
        curr_iter = i + 1
        if warmup_iters > 0 and curr_iter < warmup_iters:
            lr = get_warmup_lr(lr, curr_iter, warmup, warmup_ratio, warmup_iters)
            lrs.append(lr)
        else:
            curr_epoch = curr_iter // steps_per_epoch
            lrs.append(annealing_cos(lr, target_lr, curr_epoch / epochs))
    return lrs
