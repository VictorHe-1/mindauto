import math
import mindspore as ms
from mindspore import nn, ops
from typing import Optional
import mindspore.common.initializer as init
from mindspore.ops.function.nn_func import (
    _check_qkv_shape, _check_kpm_shape, _check_attn_mask_shape,
    _in_projection_packed, _in_projection, _inner_pad, linear, _inner_dropout
)
import mindspore.common.dtype as mstype


class _Linear(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True):
        fan_in, _ = init._calculate_fan_in_and_fan_out((out_channels, in_channels))
        bound = 1 / math.sqrt(fan_in)
        super().__init__(in_channels, out_channels, weight_init=init.HeUniform(math.sqrt(5)),
                         bias_init=init.Uniform(bound), has_bias=has_bias, activation=None)


def _scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, is_training):
    """scaled dot product attention"""
    embed_size = query.shape[-1]
    scaling_factor = ms.Tensor(embed_size, mstype.float16).sqrt().sqrt()
    query = query / scaling_factor

    if is_causal:
        L = query.shape[-2]
        S = key.shape[-2]
        attn_mask = ops.ones((L, S), mstype.bool_).tril()

    attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = ops.softmax(attn, -1)
    attn = _inner_dropout(attn, dropout_p, is_training)
    output = ops.matmul(attn, value)

    return (output, attn)


def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight,
                                 in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                                 out_proj_bias, training=True, key_padding_mask=None, attn_mask=None,
                                 use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
                                 v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True,
                                 is_causal=False, k_is_v=False, q_is_k=False):
    """multi head attetion forward function"""
    is_batched = _check_qkv_shape(query.ndim, key.ndim, value.ndim)
    if key_padding_mask is not None:
        _check_kpm_shape(query.ndim, key_padding_mask.ndim)
    if attn_mask is not None:
        _check_attn_mask_shape(query.ndim, query.shape, key.shape, attn_mask.ndim,
                               attn_mask.shape, num_heads)

    if not is_batched:
        query = query.expand_dims(1)
        key = key.expand_dims(1)
        value = value.expand_dims(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.expand_dims(0)

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != mstype.bool_ and not ops.is_floating_point(key_padding_mask):
            raise ValueError("The `key_padding_mask` only supports bool and floating dtypes.")
    if embed_dim != embed_dim_to_check:
        raise ValueError(f"The `embed_dim` should be {embed_dim_to_check}, but got {embed_dim}.")

    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(f"The `embed_dim` {embed_dim} can not be divisible by `num_heads` {num_heads}.")
    if use_separate_proj_weight:
        # allow MHA to have different embedding dims when separate projection weights are used
        if key.shape[:2] != value.shape[:2]:
            raise ValueError(f"The sequence length and batch dims of `key`: {key.shape[:2]} do not match "
                             f"`value`: {value.shape[:2]}.")
    else:
        if key.shape != value.shape:
            raise ValueError(f"The shape of `key` {key.shape} does not match `value` {value.shape}.")

    # compute in-projection
    if not use_separate_proj_weight:
        if in_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``False`` but `in_proj_weight` got ``None``.")
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias, k_is_v, q_is_k)
    else:
        if q_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``True`` but `q_proj_weight` got ``None``.")
        if k_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``True`` but `k_proj_weight` got ``None``.")
        if v_proj_weight is None:
            raise ValueError("`use_separate_proj_weight` is ``True`` but `v_proj_weight` got ``None``.")
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.tensor_split(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == mstype.uint8:
            attn_mask = attn_mask.astype(mstype.bool_)
        else:
            if not ops.is_floating_point(attn_mask) and attn_mask.dtype != mstype.bool_:
                raise ValueError(f"`attn_mask` only support float, byte, and bool types, "
                                 f"but got not {attn_mask.dtype}.")
        # ensure attn_mask's ndim is 3
        if attn_mask.ndim == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise ValueError(f"The shape of the `attn_mask` should be {correct_2d_size}, "
                                 f"but got {attn_mask.shape}.")
            attn_mask = attn_mask.expand_dims(0)
        elif attn_mask.ndim == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise ValueError(f"The shape of the `attn_mask` should be {correct_3d_size}, "
                                 f"but got {attn_mask.shape}.")
        else:
            raise ValueError(f"The ndim of `attn_mask` only support 2 or 3, "
                             f"but got {attn_mask.ndim}.")

    if bias_k is not None and bias_v is not None:
        if static_k is not None:
            raise ValueError("The bias_k cannot be added to static_k.")
        if static_v is not None:
            raise ValueError("The bias_v cannot be added to static_v.")
        k = ops.cat([k, bias_k.tile((1, bsz, 1))])
        v = ops.cat([v, bias_v.tile((1, bsz, 1))])
        if attn_mask is not None:
            attn_mask = _inner_pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = _inner_pad(key_padding_mask, (0, 1))
    else:
        if bias_k is not None or bias_v is not None:
            raise ValueError("The bias_k and bias_v should be ``None``"
                             "at the same time.")

    q = q.view((tgt_len, bsz * num_heads, head_dim)).swapaxes(0, 1)
    if static_k is None:
        k = k.view((k.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)
    else:
        if static_k.shape[0] != bsz * num_heads:
            raise ValueError(f"The shape[0] of `static_k` should be {bsz * num_heads}, "
                             f"but got {static_k.shape[0]}")
        if static_k.shape[2] != head_dim:
            raise ValueError(f"The shape[2] of `static_k` should be {head_dim}, "
                             f"but got {static_k.shape[2]}")
        k = static_k
    if static_v is None:
        v = v.view((v.shape[0], bsz * num_heads, head_dim)).swapaxes(0, 1)
    else:
        if static_v.shape[0] != bsz * num_heads:
            raise ValueError(f"The shape[0] of `static_v` should be {bsz * num_heads}, "
                             f"but got {static_v.shape[0]}")
        if static_v.shape[2] != head_dim:
            raise ValueError(f"The shape[2] of `static_v` should be {head_dim}, "
                             f"but got {static_v.shape[2]}")
        v = static_v

    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = ops.cat([k, ops.zeros(zero_attn_shape, dtype=k.dtype)], axis=1)
        v = ops.cat([v, ops.zeros(zero_attn_shape, dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = _inner_pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = _inner_pad(key_padding_mask, (0, 1))

    src_len = k.shape[1]

    if key_padding_mask is not None:
        if key_padding_mask.shape != (bsz, src_len):
            raise ValueError(f"The shape of `key_padding_mask` should be {(bsz, src_len)}, "
                             f"but got {key_padding_mask.shape}.")

        key_padding_mask = key_padding_mask.view((bsz, 1, 1, src_len)). \
            tile((1, num_heads, 1, 1)).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == mstype.bool_:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask + key_padding_mask

    if attn_mask is not None and attn_mask.dtype == mstype.bool_:
        new_attn_mask = ops.zeros_like(attn_mask, dtype=q.dtype)
        attn_mask = new_attn_mask.masked_fill(attn_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.shape[0] == 1:
            attn_mask = attn_mask.expand_dims(0)
        else:
            attn_mask = attn_mask.view((bsz, num_heads, -1, src_len))

    q = q.view((bsz, num_heads, tgt_len, head_dim))
    k = k.view((bsz, num_heads, src_len, head_dim))
    v = v.view((bsz, num_heads, src_len, head_dim))

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, is_causal, training)
    attn_output = attn_output.transpose(2, 0, 1, 3).view((bsz * tgt_len, embed_dim))

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view((tgt_len, bsz, attn_output.shape[1]))

    attn_output_weights = attn_output_weights.view((bsz, num_heads, tgt_len, src_len))
    if average_attn_weights:
        attn_output_weights = attn_output_weights.sum(axis=1) / num_heads

    if not is_batched:
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights


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
            self.q_proj_weight = ms.Parameter(
                init.initializer(init.XavierUniform(), (embed_dim, embed_dim), dtype=ms.float16),
                'q_proj_weight')
            self.k_proj_weight = ms.Parameter(
                init.initializer(init.XavierUniform(), (embed_dim, self.kdim), dtype=ms.float16),
                'k_proj_weight')
            self.v_proj_weight = ms.Parameter(
                init.initializer(init.XavierUniform(), (embed_dim, self.vdim), dtype=ms.float16),
                'v_proj_weight')
            self.in_proj_weight = None
        else:
            self.in_proj_weight = ms.Parameter(
                init.initializer(init.XavierUniform(), (3 * embed_dim, embed_dim), dtype=ms.float16),
                'in_proj_weight')
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if has_bias:
            self.in_proj_bias = ms.Parameter(init.initializer('zeros', (3 * embed_dim), dtype=ms.float16),
                                             'in_proj_bias')
        else:
            self.in_proj_bias = None
        self.out_proj = _Linear(embed_dim, embed_dim, has_bias=has_bias)

        if add_bias_kv:
            self.bias_k = ms.Parameter(init.initializer(init.XavierNormal(), (1, 1, embed_dim), dtype=ms.float16),
                                       'bias_k')
            self.bias_v = ms.Parameter(init.initializer(init.XavierNormal(), (1, 1, embed_dim), dtype=ms.float16),
                                       'bias_v')
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
