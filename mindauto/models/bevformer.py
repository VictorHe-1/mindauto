import copy

import numpy as np
import mindspore as ms
from mindspore import ops

from mindauto.core.bbox.transforms import bbox3d2result
from mindauto.models.utils.grid_mask import GridMask
from mindauto.core.bbox.structures import LiDARInstance3DBoxes
from .detectors import MVXTwoStageDetector


def split_array(array):
    split_list = np.split(array, array.shape[0])
    split_list = [np.squeeze(item) for item in split_list]
    return split_list


def ms_split_array(array):
    split_list = [item.squeeze(0) for item in array]
    return split_list


def restore_img_metas(kwargs):
    # only support batch_size = 1
    # type_conversion = {'prev_bev_exists': bool, 'can_bus': np.ndarray,
    #                     'lidar2img': list, 'scene_token': str, 'box_type_3d: type}
    # type_mapping = {
    #     "<class 'mindauto.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>": LiDARInstance3DBoxes}
    key_mapping = {
        0: 'prev_bev_exists',
        1: 'can_bus',
        2: 'lidar2img',
        3: 'scene_token',
        4: 'box_type_3d',
        5: 'img_shape'
    }
    img_meta_dict = {}
    for i, value in enumerate(kwargs):
        if i < 11:
            continue
        else:
            new_i = (i - 11) % 6
            middle_key = (i - 11) // 6
            last_key = key_mapping[new_i]
            if middle_key not in img_meta_dict:
                img_meta_dict[middle_key] = {}
            if last_key in ['prev_bev_exists', 'scene_token']:
                img_meta_dict[middle_key][last_key] = value
            elif last_key == 'lidar2img':
                img_meta_dict[middle_key][last_key] = ms_split_array(ops.split(value.squeeze(0), 1))
            elif last_key == 'box_type_3d':
                img_meta_dict[middle_key][last_key] = 1  # 1 represents LiDARInstance3DBoxes
            elif last_key == 'img_shape':
                img_shape = value.squeeze(0)
                img_meta_dict[middle_key][last_key] = ms_split_array(ops.split(img_shape, 1))
            else:  # can_bus float32
                img_meta_dict[middle_key][last_key] = value.squeeze(0)
    return img_meta_dict


# TODO: modify for new key_list
def restore_img_metas_for_test(kwargs, new_args):
    # only support batch_size = 1
    # type_conversion = {'prev_bev_exists': bool, 'can_bus': np.ndarray,
    #                     'lidar2img': list, 'scene_token': str, 'box_type_3d: type}
    type_mapping = {
        "<class 'mindauto.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>": LiDARInstance3DBoxes}
    key_list = kwargs[-1].asnumpy()[0]
    img_meta_dict = {}
    for key, value in zip(key_list, kwargs[:-1]):
        if key.startswith("img_metas"):
            key_list = key.split("/")
            last_key = key_list[-1]
            if last_key in ['prev_bev_exists', 'scene_token']:
                img_meta_dict[last_key] = value.asnumpy().item()
            elif last_key == 'lidar2img':
                img_meta_dict[last_key] = split_array(value.asnumpy()[0])
            elif last_key == 'box_type_3d':
                img_meta_dict[last_key] = type_mapping[value.asnumpy().item()]
            elif last_key == 'img_shape':
                img_shape = value.asnumpy()[0]
                img_meta_dict[last_key] = [tuple(each) for each in img_shape]
            else:  # can_bus
                img_meta_dict[last_key] = value.asnumpy()[0]
        else:
            if key == 'img':
                new_args[key] = [value.squeeze(0)]
    new_args['img_metas'] = [[img_meta_dict]]


@ms.jit
def restore_3d_bbox(kwargs):
    tensor = kwargs[9].squeeze(0)
    gravity_center = kwargs[10].squeeze(0)
    lidar_inst = {'gravity_center': gravity_center, 'tensor': tensor}
    return lidar_inst


class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_bev = None  # ops.zeros() replace None
        self.scene_token = None
        self.prev_pos = 0
        self.prev_angle = 0

    def init_weights(self):
        self.pts_bbox_head.init_weights()

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.shape[0]
        # (2, 6, 3, 480, 800), (1, 6, 3, 480, 800)
        if img is not None:
            if img.ndim == 5 and img.shape[0] == 1:
                img = ops.squeeze(img)
            elif img.ndim == 5 and img.shape[0] > 1:  # deal with batch_size > 1
                B, N, C, H, W = img.shape
                img = img.reshape(B * N, C, H, W)
            img_feats = self.img_backbone(img)  # mean abs diff 0.02
            # if len_queue == 2:
            #     img_feats = (ms.Tensor(np.random.random((12, 2048, 15, 25)), ms.float32), )
            # else:
            #     img_feats = (ms.Tensor(np.random.random((6, 2048, 15, 25)), ms.float32),)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)  # mean abs diff 0.015
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          indexes=None,
                          reference_points_cam=None,
                          bev_mask=None,
                          shift=None,
                          gt_labels_mask=None):
        """Forward function'
        Args:
            pts_feats (list[mindspore.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[mindspore.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[mindspore.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (mindspore.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, indexes, reference_points_cam, bev_mask, shift)
        losses = self.pts_bbox_head.loss(gt_bboxes_3d,
                                         gt_labels_3d,
                                         outs,
                                         img_metas,
                                         gt_labels_mask)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def construct(self,
                  *args,
                  **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        mindspore.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[mindspore.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if self.training:
            img_meta_dict = restore_img_metas(args)
            lidar_inst = restore_3d_bbox(args)
            return self.forward_train(img_metas=[img_meta_dict],
                                      gt_bboxes_3d=[lidar_inst],
                                      gt_labels_3d=[args[0].squeeze(0)],
                                      img=args[1],
                                      gt_labels_mask=args[2][0],
                                      grid_mask_img=args[3],
                                      proposals=None,
                                      gt_bboxes_ignore=None,
                                      img_depth=None,
                                      img_mask=None,
                                      indexes=args[5][0],
                                      reference_points_cam=args[6][0],
                                      bev_mask=args[7][0],
                                      shift=args[8][0])
        else:
            new_args = {}
            new_args['rescale'] = True
            restore_img_metas_for_test(args, new_args)
            return self.forward_test(**new_args)

    # def construct(self,
    #               *args,
    #               ):
    #     """Calls either forward_train or forward_test depending on whether
    #     return_loss=True.
    #     Note this setting will change the expected inputs. When
    #     `return_loss=True`, img and img_metas are single-nested (i.e.
    #     mindspore.Tensor and list[dict]), and when `resturn_loss=False`, img and
    #     img_metas should be double nested (i.e.  list[mindspore.Tensor],
    #     list[list[dict]]), with the outer list indicating test time
    #     augmentations.
    #     """
    #     args = args[0]
    #     new_args = {}
    #     if self.training:
    #         img_meta_dict = restore_img_metas(args)
    #         lidar_inst = restore_3d_bbox(args)
    #         new_args['img'] = args[:-1][1]
    #         new_args['gt_labels_mask'] = args[2][0]
    #         new_args['grid_mask_img'] = args[:-1][3]
    #         new_args['img_metas'] = [img_meta_dict]
    #         new_args['gt_labels_3d'] = [args[:-1][0].squeeze(0)]
    #         new_args['gt_bboxes_3d'] = [lidar_inst]
    #         new_args['indexes'] = args[5][0]
    #         new_args['reference_points_cam'] = args[6][0]
    #         new_args['bev_mask'] = args[7][0]
    #         new_args['shift'] = args[8][0]
    #         return self.forward_train(**new_args)
    #     else:
    #         new_args['rescale'] = True
    #         restore_img_metas_for_test(args, new_args)
    #         return self.forward_test(**new_args)

    def obtain_history_bev(self, imgs_queue, img_metas_list, indexes, reference_points_cam, bev_mask, shift):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        prev_bev = ops.zeros((1, 2500, 256), ms.float32)  # [bs, bev_h*bev_w,embed_dim]
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
        ops.stop_gradient(img_feats_list[0])
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            prev_bev = self.pts_bbox_head(
                img_feats, img_metas, prev_bev, indexes[i], reference_points_cam[i], bev_mask[i], shift[i], only_bev=True)
            prev_bev = ops.stop_gradient(prev_bev)
        return prev_bev

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      gt_labels_mask=None,
                      grid_mask_img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      indexes=None,
                      reference_points_cam=None,
                      bev_mask=None,
                      shift=None):
        """Forward training function.
        Args:
            points (list[mindspore.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[mindspore.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[mindspore.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[mindspore.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (mindspore.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[mindspore.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[mindspore.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        len_queue = img.shape[1]  # len_queue = 3
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]
        prev_grid_img = grid_mask_img[:, :-1, ...]
        grid_img = grid_mask_img[:, -1, ...]
        # img_metas: List[Dict{0: {}, 1: {}, 2: {}}]
        prev_img_metas = img_metas  # graph mode doesn't support copy.deepcopy
        if self.use_grid_mask:
            prev_bev = self.obtain_history_bev(prev_grid_img,
                                               prev_img_metas,
                                               indexes,
                                               reference_points_cam,
                                               bev_mask,
                                               shift)
        else:
            prev_bev = self.obtain_history_bev(prev_img,
                                               prev_img_metas,
                                               indexes,
                                               reference_points_cam,
                                               bev_mask,
                                               shift)
        img_metas = [each[len_queue - 1] for each in img_metas]
        if self.use_grid_mask:
            img_feats = self.extract_feat(img=grid_img, img_metas=img_metas)
        else:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
        prev_bev_exists = img_metas[0]['prev_bev_exists'].astype(ms.int32)
        true_prev_bev = prev_bev_exists * prev_bev + (1 - prev_bev_exists) * ops.zeros((1, 2500, 256), ms.float32)
        losses_pts = self.forward_pts_train(img_feats,
                                            gt_bboxes_3d,
                                            gt_labels_3d,
                                            img_metas,
                                            gt_bboxes_ignore,
                                            true_prev_bev,
                                            indexes[-1],
                                            reference_points_cam[-1],
                                            bev_mask[-1],
                                            shift[-1],
                                            gt_labels_mask[-1])
        return losses_pts

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.scene_token:
            # the first sample of each scene is truncated
            self.prev_bev = None
        # update idx
        self.scene_token = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_bev = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_bev is not None:  # self.prev_bev is not None
            img_metas[0][0]['can_bus'][:3] -= self.prev_pos
            img_metas[0][0]['can_bus'][-1] -= self.prev_angle
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_bev, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_pos = tmp_pos
        self.prev_angle = tmp_angle
        self.prev_bev = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)  # TODO
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
