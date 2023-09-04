from enum import IntEnum, unique


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
