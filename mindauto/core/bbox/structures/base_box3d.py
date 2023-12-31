from abc import abstractmethod

import numpy as np
import mindspore as ms
from mindspore import ops

from .utils import limit_period, xywhr2xyxyr


class BaseInstance3DBoxes(object):
    """Base class for 3D Boxes.

    Note:
        The box is bottom centered, i.e. the relative position of origin in
        the box is (0.5, 0.5, 0).

    Args:
        tensor (ms.Tensor | np.ndarray | list): a N x box_dim matrix.
        box_dim (int): Number of the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw).
            Default to 7.
        with_yaw (bool): Whether the box is with yaw rotation.
            If False, the value of yaw will be set to 0 as minmax boxes.
            Default to True.
        origin (tuple[float]): The relative position of origin in the box.
            Default to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.

    Attributes:
        tensor (ms.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0), numpy_boxes=False):
        self.numpy_boxes = numpy_boxes
        if self.numpy_boxes:
            self.numpy_init(tensor, box_dim, with_yaw, origin)
        else:
            self.ms_init(tensor, box_dim, with_yaw, origin)

    def numpy_init(self, tensor, box_dim, with_yaw, origin):
        self.input_tensor = np.copy(tensor)
        self.input_box_dim = box_dim
        self.input_with_yaw = with_yaw
        self.input_origin = origin
        tensor = np.array(tensor, dtype=np.float32)
        if np.size(tensor) == 0:
            tensor = tensor.reshape((0, box_dim)).astype(np.float32)
        assert tensor.ndim == 2 and tensor.shape[-1] == box_dim, tensor.shape

        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = np.zeros((tensor.shape[0], 1), dtype=tensor.dtype)
            tensor = np.concatenate((tensor, fake_rot), axis=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = np.copy(tensor)

        if origin != (0.5, 0.5, 0):
            dst = np.array((0.5, 0.5, 0), dtype=self.tensor.dtype)
            src = np.array(origin, dtype=self.tensor.dtype)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def ms_init(self, tensor, box_dim, with_yaw, origin):
        self.input_tensor = tensor.copy()
        self.input_box_dim = box_dim
        self.input_with_yaw = with_yaw
        self.input_origin = origin
        tensor = ms.Tensor(tensor, dtype=ms.float32)
        if ops.numel(tensor) == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).astype(ms.float32)
        assert tensor.ndim == 2 and tensor.shape[-1] == box_dim, tensor.shape

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # 0 as a fake yaw and set with_yaw to False.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = ops.cat((tensor, fake_rot), axis=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.copy()

        if origin != (0.5, 0.5, 0):
            dst = ms.Tensor((0.5, 0.5, 0), dtype=self.tensor.dtype)
            src = ms.Tensor(origin, dtype=self.tensor.dtype)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def volume(self):
        """ms.Tensor: A vector with volume of each box."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self):
        """ms.Tensor: Corners of each box with size (N, 8, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """ms.Tensor: A vector with yaw of each box."""
        return self.tensor[:, 6]

    @property
    def height(self):
        """ms.Tensor: A vector with height of each box."""
        return self.tensor[:, 5]

    @property
    def top_height(self):
        """ms.Tensor: A vector with the top height of each box."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self):
        """ms.Tensor: A vector with bottom's height of each box."""
        return self.tensor[:, 2]

    @property
    def center(self):
        """Calculate the center of all the boxes.

        Note:
            In the MMDetection3D's convention, the bottom center is
            usually taken as the default center.

            The relative position of the centers in different kinds of
            boxes are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
            It is recommended to use ``bottom_center`` or ``gravity_center``
            for more clear usage.

        Returns:
            ms.Tensor: A tensor with center of each box.
        """
        return self.bottom_center

    @property
    def bottom_center(self):
        """ms.Tensor: A tensor with center of each box."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self):
        """ms.Tensor: A tensor with center of each box."""
        pass

    @property
    def corners(self):
        """ms.Tensor: a tensor with 8 corners of each box."""
        pass

    @abstractmethod
    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or \
        rotation matrix.

        Args:
            angle (float | ms.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (ms.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.
        """
        pass

    @abstractmethod
    def flip(self, bev_direction='horizontal'):
        """Flip the boxes in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (ms.Tensor): Translation vector of size 1x3.
        """
        if not self.numpy_boxes and not isinstance(trans_vector, ms.Tensor):
            trans_vector = ms.Tensor(trans_vector, dtype=self.tensor.dtype)
        if self.numpy_boxes and not isinstance(trans_vector, np.ndarray):
            trans_vector = np.array(trans_vector, dtype=self.tensor.dtype)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | ms.Tensor): The range of box
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            ms.Tensor: A binary vector indicating whether each box is \
                inside the reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > box_range[0])
                          & (self.tensor[:, 1] > box_range[1])
                          & (self.tensor[:, 2] > box_range[2])
                          & (self.tensor[:, 0] < box_range[3])
                          & (self.tensor[:, 1] < box_range[4])
                          & (self.tensor[:, 2] < box_range[5]))
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, box_range):
        """Check whether the boxes are in the given range.

        Args:
            box_range (list | ms.Tensor): The range of box
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            ms.Tensor: Indicating whether each box is inside \
                the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | ms.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor

    def limit_yaw(self, offset=0.5, period=np.pi):
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw.
            period (float): The expected period.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period, self.numpy_boxes)

    def nonempty(self, threshold: float = 0.0):
        """Find boxes that are non-empty.

        A box is considered empty,
        if either of its side is no larger than threshold.

        Args:
            threshold (float): The threshold of minimal sizes.

        Returns:
            ms.Tensor: A binary vector which represents whether each \
                box is empty (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]  # ms2.1 pynative support ok
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`:
                return a `Boxes` that contains only one box.
            2. `new_boxes = boxes[2:10]`:
                return a slice of boxes.
            3. `new_boxes = boxes[vector]`:
                where vector is a ms.BoolTensor with `length = len(boxes)`.
                Nonzero elements in the vector will be selected.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of  \
                :class:`BaseInstances3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].reshape(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw,
                numpy_boxes=self.numpy_boxes)
        b = self.tensor[item]
        assert b.ndim == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw, numpy_boxes=self.numpy_boxes)

    def __len__(self):
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list, numpy_boxes):
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (list[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated Boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0 and not numpy_boxes:
            return cls(ms.numpy.empty(0))
        if len(boxes_list) == 0 and numpy_boxes:
            return cls(np.empty((0, 0)))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use ms.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        if numpy_boxes:
            cat_boxes = cls(
                np.concatenate([b.tensor for b in boxes_list], axis=0),
                box_dim=boxes_list[0].tensor.shape[1],
                with_yaw=boxes_list[0].with_yaw,
                numpy_boxes=numpy_boxes)
        else:
            cat_boxes = cls(
                ops.cat([b.tensor for b in boxes_list], dim=0),
                box_dim=boxes_list[0].tensor.shape[1],
                with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties \
                as self.
        """
        original_type = type(self)
        if self.numpy_boxes:
            return original_type(
                np.copy(self.tensor), box_dim=self.box_dim, with_yaw=self.with_yaw, numpy_boxes=self.numpy_boxes)
        else:
            return original_type(
                self.tensor.copy(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time.

        Returns:
            ms.Tensor: A box of shape (4,).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1, boxes2, mode='iou'):
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between boxes1 and
            boxes2,  boxes1 and boxes2 should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str, optional): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            ms.Tensor: Calculated iou of boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
                                             f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'
        if boxes1.numpy_boxes and boxes2.numpy_boxes:
            boxes1_top_height = boxes1.top_height.reshape(-1, 1)
            boxes1_bottom_height = boxes1.bottom_height.reshape(-1, 1)
            boxes2_top_height = boxes2.top_height.reshape(1, -1)
            boxes2_bottom_height = boxes2.bottom_height.reshape(1, -1)

            heighest_of_bottom = np.maximum(boxes1_bottom_height, boxes2_bottom_height)
            lowest_of_top = np.minimum(boxes1_top_height, boxes2_top_height)
            overlaps_h = np.clip(lowest_of_top - heighest_of_bottom, a_min=0, a_max=None)
        else:
            boxes1_top_height = boxes1.top_height.view(-1, 1)
            boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
            boxes2_top_height = boxes2.top_height.view(1, -1)
            boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

            heighest_of_bottom = ops.max(boxes1_bottom_height,
                                         boxes2_bottom_height)
            lowest_of_top = ops.min(boxes1_top_height, boxes2_top_height)
            overlaps_h = ops.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    # @classmethod
    # def overlaps(cls, boxes1, boxes2, mode='iou'):
    #     """Calculate 3D overlaps of two boxes.
    #
    #     Note:
    #         This function calculates the overlaps between ``boxes1`` and
    #         ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.
    #
    #     Args:
    #         boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
    #         boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
    #         mode (str, optional): Mode of iou calculation. Defaults to 'iou'.
    #
    #     Returns:
    #         ms.Tensor: Calculated iou of boxes' heights.
    #     """
    #     assert isinstance(boxes1, BaseInstance3DBoxes)
    #     assert isinstance(boxes2, BaseInstance3DBoxes)
    #     assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
    #         f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'
    #
    #     assert mode in ['iou', 'iof']
    #
    #     rows = len(boxes1)
    #     cols = len(boxes2)
    #     if rows * cols == 0:
    #         return ms.zeros((rows, cols), dtype=boxes1.tensor.dtype)  # boxes1.tensor.new(rows, cols)
    #
    #     # height overlap
    #     overlaps_h = cls.height_overlaps(boxes1, boxes2)
    #
    #     # obtain BEV boxes in XYXYR format
    #     boxes1_bev = xywhr2xyxyr(boxes1.bev)
    #     boxes2_bev = xywhr2xyxyr(boxes2.bev)
    #
    #     # bev overlap
    #     # TODO: replace mmdet3d iou3d_cuda or delete this method
    #     from mmdet3d.ops.iou3d import iou3d_cuda
    #     overlaps_bev = boxes1_bev.new_zeros(
    #         (boxes1_bev.shape[0], boxes2_bev.shape[0])).cuda()  # (N, M)
    #     iou3d_cuda.boxes_overlap_bev_gpu(boxes1_bev.contiguous().cuda(),
    #                                      boxes2_bev.contiguous().cuda(),
    #                                      overlaps_bev)
    #
    #     # 3d overlaps
    #     overlaps_3d = overlaps_bev * overlaps_h
    #
    #     volume1 = boxes1.volume.view(-1, 1)
    #     volume2 = boxes2.volume.view(1, -1)
    #
    #     if mode == 'iou':
    #         # the clamp func is used to avoid division of 0
    #         iou3d = overlaps_3d / ops.clamp(
    #             volume1 + volume2 - overlaps_3d, min=1e-8)
    #     else:
    #         iou3d = overlaps_3d / ops.clamp(volume1, min=1e-8)
    #
    #     return iou3d

    def new_box(self, data):
        """Create a new box object with data.

        The new box and its tensor has the similar properties \
            as self and self.tensor, respectively.

        Args:
            data (ms.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, \
                the object's other properties are similar to ``self``.
        """
        if self.numpy_boxes:
            new_tensor = np.array(data, dtype=self.tensor.dtype) \
                if not isinstance(data, np.ndarray) else data
        else:
            new_tensor = ms.Tensor(data, dtype=self.tensor.dtype) \
                if not isinstance(data, ms.Tensor) else data
        original_type = type(self)
        return original_type(
            new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw, numpy_boxes=self.numpy_boxes)
