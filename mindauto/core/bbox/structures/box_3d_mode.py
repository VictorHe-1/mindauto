from enum import IntEnum, unique
import numpy as np
import mindspore as ms
from mindspore import ops
from .base_box3d import BaseInstance3DBoxes
from .depth_box3d import DepthInstance3DBoxes
from .lidar_box3d import LiDARInstance3DBoxes


@unique
class Box3DMode(IntEnum):
    r"""Enum of different ways to represent a box.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is [0.5, 1.0, 0.5],
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth mode:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(box, src, dst, rt_mat=None):
        """Convert boxes from `src` mode to `dst` mode.

        Args:
            box (tuple | list | np.ndarray |
                torch.Tensor | BaseInstance3DBoxes):
                Can be a k-tuple, k-list or an Nxk array/tensor, where k = 7.
            src (:obj:`Box3DMode`): The src Box mode.
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            (tuple | list | np.ndarray | torch.Tensor | BaseInstance3DBoxes): \
                The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                'Box3DMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = ms.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = ms.Tensor(np.asarray(box)).copy()
            elif is_Instance3DBoxes:
                arr = box.tensor.copy()
            else:
                arr = box.copy()

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = ms.Tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=arr.dtype)
            xyz_size = ops.cat([y_size, z_size, x_size], axis=-1)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = ms.Tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=arr.dtype)
            xyz_size = ops.cat([z_size, x_size, y_size], axis=-1)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = ms.Tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=arr.dtype)
            xyz_size = ops.cat([x_size, z_size, y_size], axis=-1)
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = ms.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=arr.dtype)
            xyz_size = ops.cat([x_size, z_size, y_size], axis=-1)
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = ms.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=arr.dtype)
            xyz_size = ops.cat([y_size, x_size, z_size], axis=-1)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = ms.Tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=arr.dtype)
            xyz_size = ops.cat([y_size, x_size, z_size], axis=-1)
        else:
            raise NotImplementedError(
                f'Conversion from Box3DMode {src} to {dst} '
                'is not supported yet')

        if not isinstance(rt_mat, ms.Tensor):
            rt_mat = ms.Tensor(rt_mat, dtype=arr.dtype)
        if rt_mat.shape[1] == 4:
            extended_xyz = ops.cat(
                [arr[:, :3], arr.new_ones((arr.shape[0], 1))], axis=-1)
            xyz = ops.matmul(extended_xyz, rt_mat.t())
        else:
            xyz = ops.matmul(arr[:, :3], rt_mat.t())

        remains = arr[..., 6:]
        arr = ops.cat([xyz[:, :3], xyz_size, remains], axis=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(ops.flatten(arr).tolist())
        if is_numpy:
            return arr.asnumpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            else:
                raise NotImplementedError(
                    f'Conversion to {dst} through {original_type}'
                    ' is not supported yet')
            return target_type(
                arr, box_dim=arr.shape[-1], with_yaw=box.with_yaw)
        else:
            return arr
