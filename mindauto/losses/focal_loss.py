import mindspore as ms
from mindspore import ops, nn

from .utils import weight_reduce_focal_loss


class FocalLoss(nn.Cell):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def py_sigmoid_focal_loss(self,
                              pred,
                              target,
                              weight=None,
                              avg_factor=None
                              ):
        """Mindspore version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

        Args:
            pred (ms.Tensor): The prediction with shape (N, C), C is the
                number of classes
            target (ms.Tensor): The learning label of the prediction.
            weight (ms.Tensor, optional): Sample-wise loss weight.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        pred_sigmoid = ops.sigmoid(pred)
        target = target.astype(pred.dtype)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = ops.binary_cross_entropy_with_logits(
            pred, target, ops.ones_like(pred), ops.ones_like(pred), reduction='none') * focal_weight
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.shape[0] == loss.shape[0]:
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = weight.view(-1, 1)
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.shape[0], -1)
            assert weight.ndim == loss.ndim
        loss = weight_reduce_focal_loss(loss, weight, self.reduction, avg_factor)
        return loss

    def construct(self,
                  pred,
                  target,
                  weight=None,
                  avg_factor=None):
        """Forward function.

        Args:
            pred (ms.Tensor): The prediction. [num_bbox, num_classes]
            target (ms.Tensor): The learning label of the prediction. [num_bbox]
            weight (ms.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. [num_bbox]
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            ms.Tensor: The calculated loss
        """
        if self.use_sigmoid:
            num_classes = pred.shape[1]

            # Bugs of Pynative Mode: ops.one_hot(target.astype(ms.int64), num_classes + 1, 1, 0)
            target = ops.one_hot(target.astype(ms.int64), num_classes + 1, 1, 0)

            target = target[:, :num_classes]
            loss_cls = self.loss_weight * self.py_sigmoid_focal_loss(
                pred,
                target,
                weight,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
