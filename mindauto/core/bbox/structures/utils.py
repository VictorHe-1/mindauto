import numpy as np
from mindspore import ops

from .box_3d_mode import Box3DMode
from .lidar_box3d import LiDARInstance3DBoxes


def get_box_type(box_type):
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure.
            The valid value are "LiDAR", "Camera", or "Depth".

    Returns:
        tuple: Box type and box mode.
    """
    box_type_lower = box_type.lower()
    if box_type_lower == 'lidar':
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    else:
        raise ValueError('Only "box_type" of "lidar"'
                         f' are supported, got {box_type}')

    return box_type_3d, box_mode_3d


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (ms.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        ms.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - ops.floor(val / period + offset) * period


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (ms.Tensor): Points of shape (N, M, 3).
        angles (ms.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        ms.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = ops.sin(angles)
    rot_cos = ops.cos(angles)
    ones = ops.ones_like(rot_cos)
    zeros = ops.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = ops.stack([
            ops.stack([rot_cos, zeros, -rot_sin]),
            ops.stack([zeros, ones, zeros]),
            ops.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = ops.stack([
            ops.stack([rot_cos, -rot_sin, zeros]),
            ops.stack([rot_sin, rot_cos, zeros]),
            ops.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = ops.stack([
            ops.stack([zeros, rot_cos, -rot_sin]),
            ops.stack([zeros, rot_sin, rot_cos]),
            ops.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

    return ops.einsum('aij,jka->aik', points, rot_mat_T)


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (ms.Tensor): Rotated boxes in XYWHR format.

    Returns:
        ms.Tensor: Converted boxes in XYXYR format.
    """
    boxes = ops.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes
