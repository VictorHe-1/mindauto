import warnings

import numpy as np
from mindspore import nn, ops
import mindspore as ms
import mindspore.common.initializer as init

from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from .multi_head_attention import MindSporeMultiheadAttention
from ..utils import rotate
from common import build_activation_layer, build_dropout


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
                  reference_points=None,
                  spatial_shapes=None,
                  level_start_index=None,
                  img_metas=None
                  ):
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

    def init_reg_cls(self, reg_branches, cls_branches):
        self.reg_branches = reg_branches
        self.cls_branches = cls_branches
        self.decoder.init_reg_cls(reg_branches, cls_branches)

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
            img_metas=None,
            indexes=None,
            reference_points_cam=None,
            bev_mask=None,
            shift=None):
        """
        obtain bev features.
        """
        bs = mlvl_feats[0].shape[0]
        bev_queries = bev_queries.unsqueeze(1).tile((1, bs, 1))
        bev_pos = ops.flatten(bev_pos, start_dim=2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        prev_bev = prev_bev.permute(1, 0, 2)
        if self.rotate_prev_bev:
            for i in range(bs):
                rotation_angle = img_metas[i]['can_bus'][-1]
                tmp_prev_bev = prev_bev[:, i].reshape(
                    bev_h, bev_w, -1).permute(2, 0, 1)
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, self.rotate_center)
                tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                    bev_h * bev_w, 1, -1)
                prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = ops.stack([each['can_bus'] for each in img_metas], 0)
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
        spatial_shapes_tensor = ms.Tensor(spatial_shapes)
        level_start_index = ops.cat((spatial_shapes_tensor.new_zeros(
            (1,)).astype(ms.int32), spatial_shapes_tensor.prod(1).astype(ms.int32).cumsum(0)[:-1].astype(ms.int32)))
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h,
            bev_w,
            bev_pos,
            spatial_shapes,
            level_start_index,
            prev_bev,
            shift,
            img_metas,
            indexes,
            reference_points_cam,
            bev_mask
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
                  prev_bev=None,
                  img_metas=None,
                  indexes=None,
                  reference_points_cam=None,
                  bev_mask=None,
                  shift=None
                  ):
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
            img_metas=img_metas,
            indexes=indexes,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            shift=shift)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]
        query_pos, query = ops.split(
            object_query_embed, self.embed_dims, axis=1)
        query_pos = query_pos.unsqueeze(0).broadcast_to((bs, -1, -1))
        query = query.unsqueeze(0).broadcast_to((bs, -1, -1))
        reference_points = self.reference_points(query_pos)
        reference_points = ops.sigmoid(reference_points)
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
            spatial_shapes=[[bev_h, bev_w]],
            level_start_index=ms.Tensor([0], dtype=ms.float32),
            img_metas=img_metas)
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
