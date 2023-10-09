import warnings
from typing import Optional
import math

import numpy as np
from mindspore import nn, ops
import mindspore as ms
import mindspore.common.initializer as init
from mindspore.dataset.vision import Rotate
from mindspore.ops.function.nn_func import multi_head_attention_forward

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from common import build_activation_layer, build_dropout


# import torch.nn as nn
# nn.MultiheadAttention
class _Linear(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True):
        fan_in, _ = init._calculate_fan_in_and_fan_out((out_channels, in_channels))
        bound = 1 / math.sqrt(fan_in)
        super().__init__(in_channels, out_channels, weight_init=init.HeUniform(math.sqrt(5)),
                         bias_init=init.Uniform(bound), has_bias=has_bias, activation=None)


class MindSporeMultiheadAttention(nn.Cell):
    r"""
    Adopted from torch.nn.MultiheadAttention
    Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., has_bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("The init argument 'embed_dim' must be divisible by 'num_heads'.")

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = ms.Parameter(init.initializer(init.XavierUniform(), (embed_dim, embed_dim)),
                                              'q_proj_weight')
            self.k_proj_weight = ms.Parameter(init.initializer(init.XavierUniform(), (embed_dim, self.kdim)),
                                              'k_proj_weight')
            self.v_proj_weight = ms.Parameter(init.initializer(init.XavierUniform(), (embed_dim, self.vdim)),
                                              'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = ms.Parameter(init.initializer(init.XavierUniform(), (3 * embed_dim, embed_dim)),
                                               'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if has_bias:
            self.in_proj_bias = ms.Parameter(init.initializer('zeros', (3 * embed_dim)), 'in_proj_bias')
        else:
            self.in_proj_bias = None
        self.out_proj = _Linear(embed_dim, embed_dim, has_bias=has_bias)

        if add_bias_kv:
            self.bias_k = ms.Parameter(init.initializer(init.XavierNormal(), (1, 1, embed_dim)), 'bias_k')
            self.bias_v = ms.Parameter(init.initializer(init.XavierNormal(), (1, 1, embed_dim)), 'bias_v')
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.k_is_v = False
        self.q_is_k = False

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            query = kwargs.get('query')
            key = kwargs.get('key')
            value = kwargs.get('value')
        else:
            query = kwargs.get('query', args[0])
            key = kwargs.get('key', args[1])
            value = kwargs.get('value', args[2])
        self.k_is_v = key is value
        self.q_is_k = query is key
        return super().__call__(*args, **kwargs)

    def construct(self, query: ms.Tensor, key: ms.Tensor, value: ms.Tensor,
                  key_padding_mask: Optional[ms.Tensor] = None,
                  need_weights: bool = True, attn_mask: Optional[ms.Tensor] = None, average_attn_weights: bool = True):
        is_batched = query.ndim == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != ms.bool_ and not ops.is_floating_point(key_padding_mask):
                raise ValueError(
                    "only bool and floating types of key_padding_mask are supported")

        if self.batch_first and is_batched:
            # k_is_v and q_is_k preprocess in __call__ since Graph mode do not support `is`
            if self.k_is_v:
                if self.q_is_k:
                    query = key = value = query.swapaxes(1, 0)
                else:
                    query, key = [x.swapaxes(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.swapaxes(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                k_is_v=self.k_is_v, q_is_k=self.q_is_k)

        if self.batch_first and is_batched:
            attn_output = attn_output.swapaxes(1, 0)
        if need_weights:
            return attn_output, attn_output_weights
        return (attn_output,)


class MultiheadAttention(nn.Cell):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', p=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()
        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['p'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = MindSporeMultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def construct(self,
                  query,
                  key=None,
                  value=None,
                  identity=None,
                  query_pos=None,
                  key_pos=None,
                  attn_mask=None,
                  key_padding_mask=None,
                  **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.swapaxes(0, 1)
            key = key.swapaxes(0, 1)
            value = value.swapaxes(0, 1)
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.swapaxes(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class PerceptionTransformer(nn.Cell):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        from .builder import build_transformer_layer_sequence
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = ms.Parameter(np.random.randn(
            self.num_feature_levels, self.embed_dims).astype(np.float32))
        self.cams_embeds = ms.Parameter(
            np.random.randn(self.num_cams, self.embed_dims).astype(np.float32))
        self.reference_points = nn.Dense(self.embed_dims, 3)
        self.can_bus_mlp = nn.SequentialCell(
            nn.Dense(18, self.embed_dims // 2, weight_init='XavierUniform', bias_init='Zero'),
            nn.ReLU(),
            nn.Dense(self.embed_dims // 2, self.embed_dims, weight_init='XavierUniform', bias_init='Zero'),
            nn.ReLU(),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.append(nn.LayerNorm(normalized_shape=(self.embed_dims,), epsilon=1e-05))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.get_parameters():
            if p.ndim > 1:
                p.set_data(
                    init.initializer(init.XavierUniform(), p.shape, p.dtype))
        for name, m in self.cells_and_names():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

        self.level_embeds.set_data(init.initializer(init.Normal(),
                                                    self.level_embeds.shape,
                                                    self.level_embeds.dtype))
        self.cams_embeds.set_data(init.initializer(init.Normal(),
                                                   self.cams_embeds.shape,
                                                   self.cams_embeds.dtype))

        self.reference_points.weight.set_data(
            init.initializer(init.XavierUniform(),
                             self.reference_points.weight.shape,
                             self.reference_points.weight.dtype))

        self.reference_points.bias.set_data(
            init.initializer(init.Zero(),
                             self.reference_points.bias.shape,
                             self.reference_points.bias.dtype))

    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        bs = mlvl_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(1).tile((1, bs, 1))
        bev_pos = ops.flatten(bev_pos, start_dim=2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                            for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                            for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
                  np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
                  np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = ms.Tensor([shift_x, shift_y], dtype=bev_queries.dtype).permute(1, 0)  # xy, bs -> bs, xy
        concat = ops.Concat(axis=1)
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    # Warning: this rotation replace the original torchvision.transforms.functional.rotate
                    rotate = Rotate(degrees=rotation_angle, center=tuple(self.rotate_center))
                    tmp_prev_bev = ms.Tensor(rotate(tmp_prev_bev.asnumpy()), dtype=ms.float32)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    # prev_bev[:, i] = tmp_prev_bev[:, 0]
                    prev_bev = concat((prev_bev[:, :i], tmp_prev_bev[:, 0:1], prev_bev[:, i + 1:]))

        # add can bus signals
        can_bus = ms.Tensor([each['can_bus'] for each in kwargs['img_metas']], dtype=bev_queries.dtype)
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = ops.flatten(feat, start_dim=3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].astype(feat.dtype)
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].astype(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = ops.cat(feat_flatten, 2)
        spatial_shapes = ms.Tensor(spatial_shapes, dtype=ms.float32)
        level_start_index = ops.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

    def construct(self,
                  mlvl_feats,  # big difference
                  bev_queries,  # No diff
                  object_query_embed,  # No diff
                  bev_h,
                  bev_w,
                  grid_length=[0.512, 0.512],  # No diff
                  bev_pos=None,  # No Diff
                  reg_branches=None,
                  cls_branches=None,
                  prev_bev=None,
                  **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]
        query_pos, query = ops.split(
            object_query_embed, self.embed_dims, axis=1)
        query_pos = query_pos.unsqueeze(0).broadcast_to((bs, -1, -1))
        query = query.unsqueeze(0).broadcast_to((bs, -1, -1))
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=np.array([[bev_h, bev_w]]),
            level_start_index=ms.Tensor([0], dtype=ms.float32),
            **kwargs)
        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


class FFN(nn.Cell):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 **kwargs):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.SequentialCell(
                    nn.Dense(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(p=ffn_drop)))
            in_channels = feedforward_channels
        layers.append(nn.Dense(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(p=ffn_drop))
        self.layers = nn.SequentialCell(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.add_identity = add_identity

    def construct(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
