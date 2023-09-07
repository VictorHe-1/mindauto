import mindspore as ms
from mindspore import ops
from functools import partial

from mindauto.core.points import get_points_type


def apply_3d_transformation(pcd, coord_type, img_meta, reverse=False):
    """Apply transformation to input point cloud.

    Args:
        pcd (ms.Tensor): The point cloud to be transformed.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not.

    Note:
        The elements in img_meta['transformation_3d_flow']:
        "T" stands for translation;
        "S" stands for scale;
        "R" stands for rotation;
        "HF" stands for horizontal flip;
        "VF" stands for vertical flip.

    Returns:
        ms.Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype

    pcd_rotate_mat = (
        ms.Tensor(img_meta['pcd_rotation'], dtype=dtype)
        if 'pcd_rotation' in img_meta else ops.eye(
            3, dtype=dtype))

    pcd_scale_factor = (
        img_meta['pcd_scale_factor'] if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (
        ms.Tensor(img_meta['pcd_trans'], dtype=dtype)
        if 'pcd_trans' in img_meta else ops.zeros(
            (3), dtype=dtype))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
                                  img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
                                img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coord_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data ' \
                                   f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord
