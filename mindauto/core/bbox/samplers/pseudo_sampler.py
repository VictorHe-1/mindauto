import numpy as np

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
        pos_inds = np.unique(np.squeeze(np.nonzero(assign_result.gt_inds > 0))).tolist()
        neg_inds = np.unique(np.squeeze(np.nonzero(assign_result.gt_inds == 0))).tolist()
        # gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=ms.uint8)
        gt_flags = np.zeros(bboxes.shape[0], dtype=np.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
