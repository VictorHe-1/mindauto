import copy
from os import path as osp
import numpy as np
import random
from PIL import Image
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

import common
from .nuscenes_eval import NuScenesEval_custom
from .nuscenes_dataset import NuScenesDataset
from .transforms.transforms_factory import run_transforms


class GridMask:
    def __init__(self,
                 use_h,
                 use_w,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=0,
                 prob=1.,
                 training=True):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False
        self.l = None
        self.training = training

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def __call__(self, input_dict):
        x = input_dict['img'].copy()
        if np.random.rand() > self.prob or not self.training:
            input_dict['grid_mask_img'] = x
            return input_dict
        n, c, h, w = x.shape
        x = x.reshape(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(x.dtype)
        if self.mode == 1:
            mask = 1 - mask
        mask = np.tile(mask, (x.shape[0], 1, 1))
        if self.offset:
            offset = (2 * (np.random.rand(h, w) - 0.5)).astype(x.dtype)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask
        input_dict['grid_mask_img'] = x.reshape(n, c, h, w)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


class ObtainShift:
    def __init__(self,
                 grid_length,
                 bev_h,
                 bev_w,
                 use_shift=True):
        self.use_shift = use_shift
        self.grid_length = grid_length
        self.bev_h = bev_h
        self.bev_w = bev_w

    def __call__(self, input_dict):
        img_metas = [input_dict['img_metas'].copy()]

        delta_x = np.array([each['can_bus'][0]
                            for each in img_metas])
        delta_y = np.array([each['can_bus'][1]
                            for each in img_metas])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in img_metas])
        grid_length_y = self.grid_length[0]
        grid_length_x = self.grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
                  np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
                  np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = np.array([shift_x, shift_y]).transpose(1, 0)  # xy, bs -> bs, xy

        input_dict['shift'] = shift
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


class GetBevMask:
    def __init__(self,
                 bev_h,
                 bev_w,
                 pc_range,
                 num_points_in_pillar,
                 dim='3d',
                 bs=1,
                 dtype=np.float32):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.dim = dim
        self.bs = bs
        self.dtype = dtype

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, dtype=np.float32):
        if dim == '3d':
            zs = np.tile(np.linspace(0.5, Z - 0.5, num_points_in_pillar
                                     ).reshape(-1, 1, 1), (1, H, W)) / Z
            xs = np.tile(np.linspace(0.5, W - 0.5, W
                                     ).reshape(1, 1, W), (num_points_in_pillar, H, 1)) / W
            ys = np.tile(np.linspace(0.5, H - 0.5, H
                                     ).reshape(1, H, 1), (num_points_in_pillar, 1, W)) / H
            ref_3d = np.stack((xs, ys, zs), -1)
            ref_3d = np.transpose(np.transpose(ref_3d, (0, 3, 1, 2)
                                               ).reshape(ref_3d.shape[0], ref_3d.shape[3], -1), (0, 2, 1))
            ref_3d = np.repeat(ref_3d[None], bs, axis=0)
        else:
            raise ValueError("dim not equal to 3d")
        return ref_3d.astype(dtype)

    def point_sampling(self, reference_points, pc_range, img_metas):
        lidar2img = [img_metas['lidar2img']]
        lidar2img = np.asarray(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.copy()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = np.concatenate(
            (reference_points, np.ones_like(reference_points[..., :1])), axis=-1)

        reference_points = np.transpose(reference_points, (1, 0, 2, 3))
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]

        reference_points = np.reshape(reference_points, (D, B, 1, num_query, 4))
        reference_points = np.tile(reference_points, (1, 1, num_cam, 1, 1))
        reference_points = np.expand_dims(reference_points, axis=-1)

        lidar2img = np.reshape(lidar2img, (1, B, num_cam, 1, 4, 4))
        lidar2img = np.tile(lidar2img, (D, 1, 1, num_query, 1, 1))
        reference_points_cam = np.squeeze(lidar2img @ reference_points, axis=-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / np.maximum(
            reference_points_cam[..., 2:3], np.ones_like(reference_points_cam[..., 2:3]) * eps)
        reference_points_cam[..., 0] /= img_metas['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        bev_mask = np.nan_to_num(bev_mask)

        reference_points_cam = np.transpose(reference_points_cam, (2, 1, 3, 0, 4)).astype(np.float32)
        bev_mask = np.squeeze(np.transpose(bev_mask, (2, 1, 3, 0, 4)), axis=-1)
        return reference_points_cam, bev_mask

    def __call__(self, input_dict):
        img_metas = input_dict['img_metas'].copy()

        ref_3d = self.get_reference_points(
            self.bev_h, self.bev_w, self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar, dim='3d',
            bs=self.bs, dtype=self.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)

        indexes = []
        selection_matrix_list = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = np.nonzero(mask_per_img[0].sum(axis=-1))[0]

            identity_matrix = np.eye(self.bev_h * self.bev_w)
            selection_matrix = np.zeros_like(identity_matrix)
            selection_matrix[index_query_per_img] += identity_matrix[index_query_per_img]
            indexes.append(index_query_per_img)
            selection_matrix_list.append(selection_matrix)

        max_len = max([len(each) for each in indexes])
        input_dict['max_len'] = max_len
        input_dict['indexes'] = selection_matrix_list
        input_dict['reference_points_cam'] = reference_points_cam
        input_dict['bev_mask'] = bev_mask.astype(np.int32)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 pc_range,
                 num_points_in_pillar,
                 bs,
                 queue_length=4,
                 bev_size=(200, 200),
                 overlap_test=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.get_bev_mask = GetBevMask(
            pc_range=pc_range,
            num_points_in_pillar=num_points_in_pillar,
            bev_h=bev_size[0],
            bev_w=bev_size[1],
            bs=bs,
            dim='3d',
            dtype=np.float32
        )
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]

        self.obtain_shift = ObtainShift(
            grid_length=(self.real_h / bev_size[0],
                        self.real_w / bev_size[1]),
            bev_h=bev_size[0],
            bev_w=bev_size[1],
            use_shift=True
        )

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = run_transforms(input_dict, transforms=self.transforms)
            if self.use_grid_mask:
                example = self.grid_mask(example)
            example = self.get_bev_mask(example)
            example = self.obtain_shift(example)
            gt_labels_3d = example['gt_labels_3d']
            if self.filter_empty_gt and \
                    (example is None or ~(gt_labels_3d != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'] for each in queue]
        gt_mask_list = [each['gt_labels_mask'] for each in queue]
        grid_imgs_list = [each['grid_mask_img'] for each in queue]
        max_len_list = [each['max_len'] for each in queue]
        indexes_list = [each['indexes'] for each in queue]
        reference_cam_list = [each['reference_points_cam'] for each in queue]
        bev_mask_list = [each['bev_mask'] for each in queue]
        shift_list = [each['shift'] for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = np.stack(imgs_list)
        queue[-1]['gt_labels_mask'] = np.stack(gt_mask_list)
        queue[-1]['grid_mask_img'] = np.stack(grid_imgs_list)
        queue[-1]['img_metas'] = metas_map
        queue[-1]['max_len'] = np.stack(max_len_list)
        queue[-1]['indexes'] = np.stack(indexes_list)
        queue[-1]['reference_points_cam'] = np.stack(reference_cam_list)
        queue[-1]['bev_mask'] = np.stack(bev_mask_list)
        queue[-1]['shift'] = np.stack(shift_list)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    # ms.GeneratorDataset must have (numpy.ndarray, ...)
    def convert_data_to_numpy(self, data, training=True):  # add flag: training
        # convert img_metas to numpy ndarray to fit for ms.GeneratorDataset
        ordered_key = ['gt_labels_3d', 'img', 'gt_labels_mask', 'grid_mask_img', 'max_len',
                       'indexes', 'reference_points_cam', 'bev_mask', 'shift']
        data['gt_labels_3d'] = data['gt_labels_3d'].astype(np.int32)
        # convert gt_bboxes_3d (LiDARInstance3D) to numpy array
        gt_bbox_3d = data['gt_bboxes_3d']
        data['tensor'] = gt_bbox_3d.tensor
        data['gravity_center'] = gt_bbox_3d.gravity_center
        ordered_key.extend(['tensor', 'gravity_center'])
        data.pop('gt_bboxes_3d')
        queue_length = 0
        for key, value in data['img_metas'].items():
            for sub_key in ['prev_bev_exists', 'can_bus', 'lidar2img', 'scene_token', 'box_type_3d', 'img_shape']:
                new_key_list = ['img_metas', str(key), sub_key]
                new_key = "/".join(new_key_list)
                if sub_key == 'box_type_3d':
                    if training:
                        data[new_key] = np.array([1])
                    else:
                        data[new_key] = np.array(str(value[sub_key]))  # MS limit: convert object to str
                else:
                    if training and sub_key == 'scene_token':  # for training: Tensor must not be Tensor[String]
                        data[new_key] = np.array([0])
                    else:
                        data[new_key] = np.array(value[sub_key])

                ordered_key_list = ['img_metas', str(queue_length), sub_key]
                ordered_key.append("/".join(ordered_key_list))
            queue_length += 1
        data.pop('img_metas')

        numpy_data = []
        for key in ordered_key:
            if data[key].dtype == np.float64:
                data[key] = data[key].astype(np.float32)
            if data[key].dtype == np.int64:
                data[key] = data[key].astype(np.int32)
            numpy_data.append(data[key])
        # numpy_data.append(np.array(ordered_key))
        return tuple(numpy_data)

    def convert_data_to_numpy_test(self, data):
        # convert img_metas to numpy ndarray to fit for ms.GeneratorDataset
        ordered_key = ['img']
        for key, value in data['img_metas'][0].items():
            if key in ['can_bus', 'lidar2img', 'scene_token', 'box_type_3d', 'img_shape']:
                new_key_list = ['img_metas', key]
                new_key = "/".join(new_key_list)
                if key == 'box_type_3d':
                    data[new_key] = np.array(str(value))  # MS limit: convert object to str
                else:
                    data[new_key] = np.array(value)
                ordered_key.append(new_key)
        data.pop('img_metas')
        numpy_data = []
        for key in ordered_key:
            numpy_data.append(data[key])
        numpy_data.append(np.array(ordered_key))
        return tuple(numpy_data)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            data = self.prepare_test_data(idx)
            numpy_data = self.convert_data_to_numpy_test(data)
            return numpy_data
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            numpy_data = self.convert_data_to_numpy(data)
            return numpy_data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=osp.join(self.data_root, 'nuscenes'),
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = common.load_from_serialized(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail
