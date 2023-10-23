from mindspore import nn, ops

from mindauto.losses.utils import weight_reduce_loss


class L1Loss(nn.Cell):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def construct(self,
                  pred,
                  target,
                  weight=None,
                  avg_factor=None,
                  label_mask=None):
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        loss_bbox = self.loss_weight * l1_loss(
            pred, target)
        loss_bbox = loss_bbox * weight
        loss_bbox = weight_reduce_loss(loss_bbox, label_mask, reduction=self.reduction, avg_factor=avg_factor)
        return loss_bbox


def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.shape == target.shape
    loss = ops.abs(pred - target)
    return loss


if __name__ == '__main__':
    pred = ops.rand((6, 6, 6))
    target = ops.rand((6, 6, 6))
    loss_instance = L1Loss()
    loss = loss_instance(pred, target)
    print(loss)
