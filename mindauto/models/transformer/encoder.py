import copy
import warnings

import numpy as np
import mindspore as ms
from mindspore import ops

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .base_transformer import TransformerLayerSequence


class BEVFormerEncoder(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, bs=1, dtype=ms.float32):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        ref_y, ref_x = ops.meshgrid(
            ops.linspace(
                0.5, H - 0.5, H),
            ops.linspace(
                0.5, W - 0.5, W),
            indexing='ij'
        )
        ref_y = ref_y.astype(dtype)
        ref_x = ref_x.astype(dtype)
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = ops.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.tile((bs, 1, 1)).unsqueeze(2)
        return ref_2d

    # This function must use torch fp32!
    def point_sampling(self, reference_points, pc_range, img_metas):
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(ops.stack(img_meta['lidar2img'], axis=0))
        lidar2img = ops.stack(lidar2img, axis=0)  # (B, N, 4, 4)
        reference_points = reference_points.copy()

        reference_points_part1 = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points_part2 = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points_part3 = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points = ops.concat([reference_points_part1, reference_points_part2, reference_points_part3], axis=-1)

        reference_points = ops.cat(
            (reference_points, ops.ones_like(reference_points[..., :1])), axis=-1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        reference_points = reference_points.view((
            D, B, 1, num_query, 4)).tile((1, 1, num_cam, 1, 1)).unsqueeze(-1)

        lidar2img = lidar2img.view((
            1, B, num_cam, 1, 4, 4)).tile((D, 1, 1, num_query, 1, 1))

        reference_points_cam = ops.matmul(lidar2img.astype(ms.float32),
                                          reference_points.astype(ms.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / ops.maximum(
            reference_points_cam[..., 2:3], ops.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0) \
                    & (reference_points_cam[..., 1:2] < 1.0) \
                    & (reference_points_cam[..., 0:1] < 1.0) \
                    & (reference_points_cam[..., 0:1] > 0.0))

        bev_mask = bev_mask.astype(ms.float32)
        bev_mask = ops.nan_to_num(bev_mask)  # TODO: ops.nan_to_num doesn't support bool

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        return reference_points_cam, bev_mask

    def construct(self,
                  bev_query,
                  key,
                  value,
                  bev_h=None,
                  bev_w=None,
                  bev_pos=None,
                  spatial_shapes=None,
                  level_start_index=None,
                  prev_bev=None,
                  shift=0.,
                  img_metas=None,
                  indexes=None,
                  reference_points_cam=None,
                  bev_mask=None,
                  valid_ratios=None):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            shift: [1, 2]
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = bev_query
        intermediate = []

        # ref_3d = self.get_reference_points(
        #     bev_h, bev_w, self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar, dim='3d',
        #     bs=bev_query.shape[1], dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(bev_h, bev_w, bev_query.shape[1], bev_query.dtype)
        # reference_points_cam, bev_mask = self.point_sampling(
        #     ref_3d, self.pc_range, img_metas)

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.copy()
        shift_ref_2d += shift[:, None, None, :]
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        prev_bev = prev_bev.permute(1, 0, 2).astype(ms.float32)

        is_first_frame = ops.stop_gradient((prev_bev.sum() == 0).astype(ms.float32))
        value_selected = is_first_frame * prev_bev + (1 - is_first_frame) * bev_query
        ref_selected = is_first_frame * ref_2d + (1 - is_first_frame) * shift_ref_2d
        prev_bev = ops.stack([prev_bev, value_selected], 1).reshape(bs * 2, len_bev, -1)
        hybird_ref_2d = ops.stack([ref_selected, ref_2d], 1).reshape(
            bs * 2, len_bev, num_bev_level, 2)
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                bev_pos,
                hybird_ref_2d,
                bev_h,
                bev_w,
                spatial_shapes,
                level_start_index,
                reference_points_cam,
                bev_mask,
                prev_bev,
                img_metas,
                indexes)
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        if self.return_intermediate:
            return ops.stack(intermediate)

        return output


class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def construct(self,
                  query,
                  key=None,
                  value=None,
                  bev_pos=None,
                  ref_2d=None,
                  bev_h=None,
                  bev_w=None,
                  spatial_shapes=None,
                  level_start_index=None,
                  reference_points_cam=None,
                  bev_mask=None,
                  prev_bev=None,
                  img_metas=None,
                  indexes=None,
                  mask=None,
                  query_pos=None,
                  key_pos=None,
                  attn_masks=None,
                  query_key_padding_mask=None,
                  key_padding_mask=None):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, ms.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:  # ['self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm']
            if layer == 'self_attn':  # temporal_self_attention
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    bev_pos,
                    query_key_padding_mask,
                    ref_2d,
                    [[bev_h, bev_w]],
                    ms.Tensor([0]),
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    bev_mask=bev_mask,
                    img_metas=img_metas)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':  # spatial_cross_attn
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos,
                    key_padding_mask,
                    spatial_shapes,
                    reference_points_cam,
                    bev_mask,
                    level_start_index,
                    key_pos,
                    mask,
                    attn_masks[attn_index],
                    img_metas,
                    indexes)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
