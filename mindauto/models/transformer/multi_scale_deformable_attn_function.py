import mindspore as ms
from mindspore import ops

'''
TODO: If using ms.GRAPH_MODE,
please comment out this line: value_spatial_shapes = value_spatial_shapes.tolist()
And before constructing, convert value_spatial_shapes to a list using value_spatial_shapes.tolist().
'''


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (np.ndarray): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = \
        sampling_locations.shape
    # since value_spatial_shapes only has 1 level we don't need to split
    # value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
    #                          axis=1)  # value_spatial_shapes dynamic
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    value_l_ = ops.swapaxes(ops.flatten(value, start_dim=2), 1, 2).reshape(
        bs * num_heads, embed_dims, value_spatial_shapes[0][0], value_spatial_shapes[0][1])
    # bs, num_queries, num_heads, num_points, 2 ->
    # bs, num_heads, num_queries, num_points, 2 ->
    # bs*num_heads, num_queries, num_points, 2
    sampling_grid_l_ = ops.flatten(ops.swapaxes(sampling_grids[:, :, :,
                                                0], 1, 2), start_dim=0, end_dim=1)
    # bs*num_heads, embed_dims, num_queries, num_points
    sampling_grid_l_ = sampling_grid_l_.astype(ms.float16)
    sampling_value_l_ = ops.grid_sample(
        value_l_,
        sampling_grid_l_,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
    sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = ops.swapaxes(attention_weights, 1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    output = ops.sum(ops.flatten(ops.stack(sampling_value_list, axis=-2), start_dim=-2) *
                     attention_weights, -1).view(bs, num_heads * embed_dims,
                                                 num_queries)
    return ops.swapaxes(output, 1, 2)
