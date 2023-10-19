import mindspore as ms
from mindspore import ops

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


class PseudoSampler(BaseSampler):
    """
    A pseudo sampler that does not do sampling actually.
    from mmdet.core.bbox.samplers import pseudo_sampler
    """
    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (mindspore.Tensor): Bounding boxes
            gt_bboxes (mindspore.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = ops.squeeze(ops.nonzero((assign_result.gt_inds > 0).astype(ms.int32)))
        neg_inds = ops.squeeze(ops.nonzero((assign_result.gt_inds == 0).astype(ms.int32)))
        gt_flags = ops.zeros(bboxes.shape[0], dtype=ms.int32)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
