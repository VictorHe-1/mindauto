import copy

import numpy as np
import mindspore as ms
from mindspore import nn, ops
import mindspore.common.initializer as init
from mindspore.communication.management import GlobalComm

from mindauto.core.bbox.coders import build_bbox_coder
from mindauto.models.transformer import inverse_sigmoid
from mindauto.core.bbox.util import normalize_bbox
from mindauto.core.bbox.structures import LiDARInstance3DBoxes
from .detr_head import DETRHead
from .dist_utils import reduce_mean


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class BEVFormerHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is None:
            code_weights = [1.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = ms.Parameter(
            code_weights, requires_grad=False, name='code_weights')

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            cls_branches_list = []
            for i in range(num_pred):
                cls_branch = []
                for _ in range(self.num_reg_fcs):
                    cls_branch.append(nn.Dense(self.embed_dims, self.embed_dims))
                    cls_branch.append(nn.LayerNorm(normalized_shape=(self.embed_dims,), epsilon=1e-05))
                    cls_branch.append(nn.ReLU())
                cls_branch.append(nn.Dense(self.embed_dims, self.cls_out_channels))
                fc_cls = nn.SequentialCell(*cls_branch)
                cls_branches_list.append(fc_cls)

            reg_branches_list = []
            for i in range(num_pred):
                reg_branch = []
                for _ in range(self.num_reg_fcs):
                    reg_branch.append(nn.Dense(self.embed_dims, self.embed_dims))
                    reg_branch.append(nn.ReLU())
                reg_branch.append(nn.Dense(self.embed_dims, self.code_size))
                reg_branch = nn.SequentialCell(*reg_branch)
                reg_branches_list.append(reg_branch)
            self.cls_branches = nn.CellList(cls_branches_list)
            self.reg_branches = nn.CellList(reg_branches_list)
        else:
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(nn.Dense(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(normalized_shape=(self.embed_dims,), epsilon=1e-05))
                cls_branch.append(nn.ReLU())
            cls_branch.append(nn.Dense(self.embed_dims, self.cls_out_channels))
            fc_cls = nn.SequentialCell(*cls_branch)

            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(nn.Dense(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU())
            reg_branch.append(nn.Dense(self.embed_dims, self.code_size))
            reg_branch = nn.SequentialCell(*reg_branch)
            self.cls_branches = nn.CellList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.CellList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                m[-1].bias.set_data(init.initializer(bias_init, m[-1].bias.shape, m[-1].bias.dtype))

    def construct(self,
                  mlvl_feats,
                  img_metas,
                  prev_bev=None,
                  indexes=None,
                  reference_points_cam=None,
                  bev_mask=None,
                  shift=None,
                  only_bev=False,
                  ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.embedding_table.astype(dtype)
        bev_queries = self.bev_embedding.embedding_table.astype(dtype)

        new_mask = ops.zeros((bs, self.bev_h, self.bev_w)).astype(dtype)
        bev_pos = self.positional_encoding(new_mask).astype(dtype)
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                (self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos,
                prev_bev,
                img_metas,
                indexes,
                reference_points_cam,
                bev_mask,
                shift
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                prev_bev=prev_bev,
                img_metas=img_metas,
                indexes=indexes,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                shift=shift
            )
        # init reference no diff
        # inter_references abs diff: 0.001
        # hs abs big diff: 0.04 median: 0.01
        # bev_embed big diff: 0.01  median: 0.01
        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3

            tmp_part1 = tmp[..., 0:2]
            tmp_part2 = tmp[..., 4:5]
            reference_part1 = reference[..., 0:2]
            reference_part2 = reference[..., 2:3]
            tmp_part1 = tmp_part1 + reference_part1
            tmp_part2 = tmp_part2 + reference_part2
            tmp_part1 = ops.sigmoid(tmp_part1)
            tmp_part2 = ops.sigmoid(tmp_part2)
            new_tmp_part1 = (tmp_part1[..., 0:1] * (self.pc_range[3] -
                                                    self.pc_range[0]) + self.pc_range[0])

            new_tmp_part2 = (tmp_part1[..., 1:2] * (self.pc_range[4] -
                                                    self.pc_range[1]) + self.pc_range[1])

            new_tmp_part3 = (tmp_part2 * (self.pc_range[5] -
                                          self.pc_range[2]) + self.pc_range[2])
            tmp = ops.concat((new_tmp_part1, new_tmp_part2, tmp[..., 2:4], new_tmp_part3, tmp[..., 5:]), axis=-1)

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = ops.stack(outputs_classes)
        outputs_coords = ops.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None,
                           gt_labels_mask=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format. (350, 9)
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_gt = gt_bboxes.shape[0]
        # assigner and sampler
        assigned_cls_pred, assigned_bbox_pred, bbox_targets, labels = self.assigner(bbox_pred,
                                                                                    cls_score,
                                                                                    gt_bboxes,
                                                                                    gt_labels,
                                                                                    gt_bboxes_ignore,
                                                                                    gt_labels_mask)
        label_weights = gt_bboxes.new_ones(num_gt)  # 350
        bbox_weights = ops.ones_like(bbox_pred)[:num_gt, :]  # 350
        return labels, label_weights, bbox_targets, bbox_weights, gt_labels_mask, assigned_bbox_pred, assigned_cls_pred

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    gt_labels_mask_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        labels_list = []
        label_weights_list = []
        bbox_targets_list = []
        bbox_weights_list = []
        label_mask_list = []
        bbox_pred_list = []
        cls_pred_list = []
        for i in range(len(cls_scores_list)):
            labels, label_weights, bbox_targets, bbox_weights, gt_labels_mask, assigned_bbox_pred, assigned_cls_pred = self._get_target_single(
                cls_scores_list[i], bbox_preds_list[i], gt_labels_list[i], gt_bboxes_list[i], gt_bboxes_ignore_list[i],
                gt_labels_mask_list)
            labels_list.append(labels)
            label_weights_list.append(label_weights)
            bbox_targets_list.append(bbox_targets)
            bbox_weights_list.append(bbox_weights)
            label_mask_list.append(gt_labels_mask)
            bbox_pred_list.append(assigned_bbox_pred)
            cls_pred_list.append(assigned_cls_pred)

        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, label_mask_list, bbox_pred_list, cls_pred_list)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    gt_labels_mask_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.shape[0]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list, gt_labels_mask_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         label_mask_list, bbox_pred_list, cls_pred_list) = cls_reg_targets

        labels = ops.cat(labels_list, 0)
        label_weights = ops.cat(label_weights_list, 0)
        bbox_targets = ops.cat(bbox_targets_list, 0)
        bbox_weights = ops.cat(bbox_weights_list, 0)

        # classification loss
        assigned_cls_scores = cls_pred_list[0].reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = label_mask_list[0].sum()
        # if self.sync_cls_avg_factor and GlobalComm.INITED:  # in distribute mode
        #     cls_avg_factor = reduce_mean(
        #         ms.Tensor([cls_avg_factor], dtype=cls_scores.dtype))

        cls_avg_factor = max(cls_avg_factor, ms.Tensor(1))
        loss_cls = self.loss_cls(
            assigned_cls_scores, labels, label_weights, avg_factor=cls_avg_factor, label_mask=label_mask_list[0])  # FocalLoss

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = label_mask_list[0].sum()

        # regression L1 loss
        assigned_bbox_preds = bbox_pred_list[0].reshape(-1, bbox_preds.shape[-1])
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # isnotnan = ops.isfinite(normalized_bbox_targets).all(axis=-1)
        num_bbox = bbox_weights.shape[0]
        bbox_weights = bbox_weights * ops.tile(self.code_weights, (num_bbox, 1))

        loss_bbox = self.loss_bbox(
            assigned_bbox_preds[..., :10],
            normalized_bbox_targets[..., :10],
            bbox_weights[..., :10],
            avg_factor=num_total_pos,
            label_mask=label_mask_list[0])  # L1Loss
        loss_cls = ops.nan_to_num(loss_cls)
        loss_bbox = ops.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None,
             gt_labels_mask=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 10].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        num_dec_layers = len(all_cls_scores)

        gt_bboxes_list = [ops.cat(
            (gt_bboxes['gravity_center'], gt_bboxes['tensor'][:, 3:]),
            axis=1) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_labels_mask for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls = []
        losses_bbox = []
        for i in range(len(all_gt_bboxes_list)):
            loss_cls, loss_bbox = self.loss_single(all_cls_scores[i],
                                                   all_bbox_preds[i],
                                                   all_gt_bboxes_list[i],
                                                   all_gt_labels_list[i],
                                                   all_gt_bboxes_ignore_list[i],
                                                   all_gt_masks_list[i])
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                ops.zeros_like(ms.Tensor(gt_labels_list[i]))
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore, gt_labels_mask)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = LiDARInstance3DBoxes(bboxes, code_size)  # img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list
