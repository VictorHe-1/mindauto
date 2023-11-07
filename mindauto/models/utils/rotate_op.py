import math
from typing import List, Optional
import mindspore as ms
from mindspore import ops, Tensor
from mindspore.ops import grid_sample


def get_rotate_matrix(center_f, angle, dtype=ms.float32):
    """
    get rotation matrix according to rotation center and angle
    res = R(img - c) + c
    """
    if not ops.is_tensor(angle):
        angle = ms.Tensor(angle, dtype=dtype)
    angle_r = ms.numpy.radians(angle)
    rotation = ops.stack(
        [ops.stack([angle_r.cos(), -angle_r.sin()]),
         ops.stack([angle_r.sin(), angle_r.cos()])]
    )
    translation = (- rotation + ops.eye(2, 2, dtype)).matmul(ms.Tensor(center_f, dtype))
    translation = translation.astype(rotation.dtype)
    matrix = ops.concat([rotation, translation.reshape(2, 1)], axis=1) # (2, 3)
    return matrix


def gen_grid(matrix, h, w, dtype=ms.float32):
    d = 0.5
    x_grid = ops.linspace(ms.Tensor(-w * 0.5 + d, dtype=dtype),
                          ms.Tensor(w * 0.5 + d - 1, dtype=dtype),
                          steps=w)
    y_grid = ops.linspace(ms.Tensor(-h * 0.5 + d, dtype=dtype),
                          ms.Tensor(h * 0.5 + d - 1, dtype=dtype),
                          steps=h)
    x_mesh, y_mesh = ops.meshgrid(x_grid, y_grid)  # (h, w, 2)
    ones = ops.ones((h, w), dtype=dtype)
    base_grid = ops.stack([x_mesh, y_mesh, ones], axis=-1)  # (h, w, 3)
    rescaled_theta = ops.swapaxes(matrix, 0, 1) / ms.Tensor([0.5 * w, 0.5 * h], dtype=dtype) # (3, 2)
    output_grid = base_grid.matmul(rescaled_theta)  # (h, w, 2)
    return output_grid


def rotate(img, angle, center=Optional[List[int]], interpolation='nearest'):
    assert len(img.shape) == 3 or len(img.shape) == 4, 'image shape should be in [C, H, W] or [N, C, H, W] format'
    need_dim_up = len(img.shape) == 3

    height, width = img.shape[-2:]
    center_f = [0.0, 0.0]
    if center is not None:
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]
    matrix = get_rotate_matrix(center_f, angle) # (2, 3)
    grid = gen_grid(matrix, w=width, h=height).unsqueeze(0)

    if need_dim_up:
        img = img.unsqueeze(0)
    grid = grid.astype(img.dtype)
    sampled_img = grid_sample(img, grid, mode=interpolation, padding_mode="zeros", align_corners=False)
    # aa = torch.nn.functional.grid_sample(torch.from_numpy(img.asnumpy()), torch.from_numpy(grid.asnumpy()), mode=interpolation, padding_mode="zeros", align_corners=False)
    if need_dim_up:
        sampled_img = sampled_img.squeeze(0)

    return sampled_img
