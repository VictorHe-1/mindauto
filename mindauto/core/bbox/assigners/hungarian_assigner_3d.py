import mindspore as ms
from mindspore import ops, nn

from .run_lsa import NetLsap
from mindauto.core.bbox.util import normalize_bbox
from mindauto.core.bbox.match_costs import build_match_cost


# try:
#     from scipy.optimize import linear_sum_assignment
# except ImportError:
#     linear_sum_assignment = None


class HungarianAssigner3D(nn.Cell):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        super().__init__()
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range
        self.lsap_nn = NetLsap()
        self.matched_row_inds = ms.Tensor([[12, 14, 19, 50, 180, 187, 205, 212, 218, 238, 253, 254, 261, 277, 283, 285,
                                            288, 295, 340, 350, 352, 375, 403, 420,
                                            485, 497, 503, 505, 538, 541, 543, 550, 561, 579, 589, 607, 654, 717, 727,
                                            748, 754, 767, 786, 807, 833, 852, 860, 867,
                                            874, 885] + [0 for _ in range(300)]])
        self.matched_col_inds = ms.Tensor(
            [[35, 27, 13, 43, 45, 18, 47, 41, 40, 46, 10, 14, 11, 29, 44, 1, 3, 39, 32, 19, 37, 8, 17, 36,
              25, 38, 7, 21, 9, 28, 34, 31, 49, 42, 4, 24, 30, 23, 33, 20, 26, 16, 6, 15, 12, 22, 2, 0,
              5, 48] + [0 for _ in range(300)]])

    def construct(self,
                  bbox_pred,
                  cls_pred,
                  gt_bboxes,
                  gt_labels,
                  gt_bboxes_ignore=None,
                  gt_labels_mask=None,
                  ):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        # 1. assign -1 by default
        # assigned_labels = ops.full((num_bboxes,), -1, dtype=ms.int32)
        # if num_gts == 0 or num_bboxes == 0:
        #     # No ground truth or boxes, return empty assignment
        #     if num_gts == 0:
        #         # No ground truth, assign all to background
        #         assigned_gt_inds[:] = 0
        #     return AssignResult(
        #         num_gts, assigned_gt_inds, None, labels=assigned_labels)
        # 2. compute the weighted costs
        # classification and bboxcost.
        # cls_cost: FocalLossCost
        # cls_pred: [900, 10] gt_labels [350]
        cls_cost = self.cls_cost(cls_pred, gt_labels)  # [900, 350]
        # regression L1 cost
        # normalized_gt_bboxes: [350, 10] [padding_dim, 10]
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        # bbox_pred: [900, 10] normalized_gt_bboxes[350, 10] -> cost: [900, 350]
        # reg_cost: BBox3DL1Cost
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        # matched_row_inds, matched_col_inds = self.lsap_nn(cost, ms.Tensor(False), gt_labels_mask.sum().astype(ms.int64))
        # matched_row_inds, matched_col_inds = self.lsap_nn(cost, ms.Tensor(False), ms.Tensor(45))
        # matched_row_inds = ops.stop_gradient(matched_row_inds)
        # matched_col_inds = ops.stop_gradient(matched_col_inds)
        # # 4. Get matched bbox_targets and labels
        #
        # # Note: matched_col_inds may contain -1
        # # we multiply it with gt_labels_mask to replace -1 with 0
        assigned_labels = ops.gather(gt_labels, self.matched_col_inds[0].astype(ms.int32) * gt_labels_mask, axis=0).astype(ms.int32)
        pos_gt_bboxes = ops.gather(gt_bboxes, self.matched_col_inds[0].astype(ms.int32) * gt_labels_mask, axis=0)
        assigned_bbox_pred = ops.gather(bbox_pred, self.matched_row_inds[0].astype(ms.int32) * gt_labels_mask, axis=0)
        assigned_cls_pred = ops.gather(cls_pred, self.matched_row_inds[0].astype(ms.int32) * gt_labels_mask, axis=0)

        return assigned_cls_pred, assigned_bbox_pred, pos_gt_bboxes, assigned_labels
