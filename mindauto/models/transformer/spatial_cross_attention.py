import warnings
import math

import mindspore as ms
from mindspore import nn, ops
import mindspore.common.initializer as init

from .multi_scale_deformable_attn_function import multi_scale_deformable_attn_pytorch


class MSDeformableAttention3D(nn.Cell):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Dense(
            embed_dims, num_heads * num_levels * num_points * 2,
            weight_init='Zero', bias_init='Zero')
        self.attention_weights = nn.Dense(embed_dims,
                                          num_heads * num_levels * num_points,
                                          weight_init='Zero',
                                          bias_init='Zero')
        self.value_proj = nn.Dense(embed_dims, embed_dims, weight_init='XavierUniform', bias_init='Zero')

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        thetas = ops.arange(
            self.num_heads,
            dtype=ms.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = ops.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdims=True)[0]).view(
            self.num_heads, 1, 1,
            2).tile((1, self.num_levels, self.num_points, 1))

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.set_data(grid_init.view(-1))
        self._is_init = True

    def construct(self,
                  query,
                  key=None,
                  value=None,
                  reference_points=None,
                  spatial_shapes=None,
                  level_start_index=None,
                  identity=None,
                  query_pos=None,
                  key_padding_mask=None,
                  ):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        sampling_locations = None
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert ops.prod(spatial_shapes) == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = nn.Softmax()(attention_weights)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = ops.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], axis=-1).astype(ms.float32)
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = ops.expand_dims(reference_points, axis=2)
            reference_points = ops.expand_dims(reference_points, axis=3)
            reference_points = ops.expand_dims(reference_points, axis=4)

            # graph mode doesn't support offset_normalizer[None, None, None, :, None, :]
            offset_normalizer = ops.expand_dims(offset_normalizer, axis=1)
            offset_normalizer = ops.expand_dims(offset_normalizer, axis=0)
            offset_normalizer = ops.expand_dims(offset_normalizer, axis=0)
            offset_normalizer = ops.expand_dims(offset_normalizer, axis=0)

            sampling_offsets = sampling_offsets / offset_normalizer
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class SpatialCrossAttention(nn.Cell):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__()
        from .builder import build_attention
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(p=dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Dense(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        self.output_proj.weight.set_data(
            init.initializer(init.XavierUniform(), self.output_proj.weight.shape, self.output_proj.weight.dtype))
        self.output_proj.bias.set_data(
            init.initializer(init.Zero(), self.output_proj.bias.shape, self.output_proj.bias.dtype))

    def construct(self,
                  query,
                  key,
                  value,
                  residual=None,
                  query_pos=None,
                  key_padding_mask=None,
                  spatial_shapes=None,
                  reference_points_cam=None,
                  bev_mask=None,
                  level_start_index=None,
                  key_pos=None,
                  mask=None,
                  attn_mask=None,
                  img_metas=None,
                  indexes=None,
                  flag='encoder'):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        inp_residual = None
        slots = None

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = ops.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.shape[3]
        max_len = 2500  # bev_h * bev_w

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch_bs = []
        reference_points_rebatch_bs = []
        for i, reference_points_per_img in enumerate(reference_points_cam):
            index_query_per_img = indexes[i]
            queries_rebatch_bs.append(ops.matmul(index_query_per_img, query[0]))
            reference_points_rebatch_bs.append(
                ops.matmul(index_query_per_img, reference_points_per_img[0].reshape(max_len, D * 2)).reshape(max_len, D,
                                                                                                             2)
            )
        queries_rebatch = ops.expand_dims(ops.stack(queries_rebatch_bs), 0)  # [bs, self.num_cams, max_len, self.embed_dims]
        reference_points_rebatch = ops.expand_dims(ops.stack(reference_points_rebatch_bs), 0)  # [bs, self.num_cams, max_len, D, 2]
        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                            key,
                                            value,
                                            reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2),
                                            spatial_shapes,
                                            level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j] += ops.matmul(index_query_per_img, queries[j, i])
                # slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = ops.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual
