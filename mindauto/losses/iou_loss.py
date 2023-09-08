from mindauto.losses.utils import weight_reduce_loss

from mindspore import nn, ops, Tensor
import mindspore as ms


class GIoULoss(nn.Cell):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def construct(self,
                pred,
                target,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not ops.any(weight > 0):
            if pred.ndim == weight.ndim + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.ndim > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(pred, target, eps=self.eps)
        loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


def giou_loss(pred, target, eps: float = 1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes."""

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new_zeros(batch_shape + (rows, ))
        else:
            return bboxes1.new_zeros(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = ops.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = ops.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = ops.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = ops.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = ops.maximum(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = ops.minimum(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = ops.minimum(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = ops.maximum(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = Tensor([eps], dtype=union.dtype)
    union = ops.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = ops.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def fp16_clamp(x, min=None, max=None):
    if x.dtype == ms.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        x = ops.cast(x, ms.float32)
        x = ops.clamp(x, min, max)
        x = ops.cast(x, ms.float16)
        return x

    return ops.clamp(x, min, max)


if __name__ == '__main__':
    pred = ops.rand((6, 4))
    target = ops.rand((6, 4))
    loss_instance = GIoULoss()
    loss = loss_instance(pred, target)
    print(loss)
