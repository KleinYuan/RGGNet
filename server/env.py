import os
import pathlib
import numpy as np
import cv2
from random import shuffle
from utils import cal_se3_error, se3_to_SE3
from src.data_raw import load_raw_data
from src.data_training import generate_single_example
from src.transform import generate_depth_map
import copy


class Env(object):
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name

    def fs_mode(self):
        return self.config

    def prepare_all_testing_data(self):
        # Parse Configs
        _raw_data_dir = self.config.fs_mode.raw_data_dir
        _num_data = self.config.fs_mode.number

        # Load all data
        print("Loading data ...")
        _all_raw_data = load_raw_data(_raw_data_dir)
        # Generate data
        _all_training_raw_data = self.get_training_raw_data(all_raw_data=_all_raw_data, num_data=_num_data)
        print("All data prepared !")
        return _all_training_raw_data

    def prepare_one_inference_data(self, example):
        if 'force_shape' in self.config.fs_mode:
            force_shape = (self.config.fs_mode.force_shape.H, self.config.fs_mode.force_shape.W)
        else:
            force_shape = None
        _range_rot = self.config.fs_mode.off_range.rot
        _range_trans = self.config.fs_mode.off_range.trans
        _vae_resized_H = self.config.fs_mode.vae_resized.H
        _vae_resized_W = self.config.fs_mode.vae_resized.W
        if 'reduce_lidar_line_to' in self.config.fs_mode:
            reduce_lidar_line_to = self.config.fs_mode.reduce_lidar_line_to
        else:
            reduce_lidar_line_to = None

        _training_data = generate_single_example(
            raw_data=example,
            range_rot_deg=_range_rot,
            range_trans_m=_range_trans,
            vae_resized_H=_vae_resized_H,
            vae_resized_W=_vae_resized_W,
            force_shape=force_shape,
            reduce_lidar_line_to=reduce_lidar_line_to)

        # TODO: This order must align with the config order, this shall be handled more carefully
        inference_data = [
            [_training_data.x_dm],
            [_training_data.x_cam / 255.]
        ]

        return inference_data, _training_data

    @staticmethod
    def get_training_raw_data(all_raw_data, num_data=False, shuffle_order=False):
        """
        Directing pre-compute all training data will make your memory explode
        So we only pre-compute the raw data s for training which is by default not loaded
        """
        print("Generating training raw data objects............")
        num_raw_data = len(all_raw_data)
        _iters = num_data // num_raw_data + 1

        training_raw_data = []
        for _ in range(0, _iters):
            for _idx, _one_raw_data in enumerate(all_raw_data):
                training_raw_data.append(_one_raw_data)

        # shuffle
        if shuffle_order:
            shuffle(training_raw_data)
        if num_data:
            return training_raw_data[:num_data]
        else:
            return training_raw_data

    def add_text(self, img, text):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (20, 30)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (250, 250, 250)
        # Line thickness of 2 px
        thickness = 2
        return cv2.putText(img, text, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

    def write_report(self, example, pred):
        _res_data_dir = self.config.fs_mode.res_data_dir + '/' + self.model_name
        _se3_true = example.y_se3param
        _se3_pred = pred[0][0]
        _se3_error = cal_se3_error(se3_pred=_se3_pred, se3_true=_se3_true)

        _SE3_pred = se3_to_SE3(_se3_pred)
        _SE3_init = se3_to_SE3(example.init_se3param)
        _SE3_corrected = np.dot(_SE3_pred, _SE3_init)

        if not os.path.isdir(_res_data_dir):
            print("{} does not exits, creating one.".format(_res_data_dir))
            pathlib.Path(_res_data_dir).mkdir(parents=True, exist_ok=True)
        # TODO: Write examples into file
        _cam = example.x_cam
        _kitti_gt_dm = example.kitti_gt_dm
        _x_dm = example.x_dm
        H, W = _cam.shape[:2]
        _x_hat_dm = generate_depth_map(
            pts_3d_in=example.pts_data,
            transform_rm_3by4=_SE3_corrected[:3, :],
            R_rect=example.x_R_rects,
            P_rect=example.x_P_rects,
            H=H,
            W=W)

        _gt_canvas = self._get_overlay(_cam, _kitti_gt_dm)
        _init_canvas = self._get_overlay(_cam, _x_dm)
        _pred_canvas = self._get_overlay(_cam, _x_hat_dm)

        _gt_canvas = self.add_text(_gt_canvas, 'GT')
        _init_canvas = self.add_text(_init_canvas, 'Input')
        _pred_canvas = self.add_text(_pred_canvas, 'Pred')
        _vis = np.concatenate([_init_canvas, _pred_canvas, _gt_canvas], 0)
        # _vis = cv2.resize(_vis, (int(_vis.shape[1]/4), int(_vis.shape[0]/4)))
        # cv2.imshow("1", _vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("{}/{}_{}.png".format(_res_data_dir, example.raw_data.id, _se3_error[0][0]), _vis)

    @staticmethod
    def asStride(arr, sub_shape, stride):
        '''Get a strided sub-matrices view of an ndarray.
        See also skimage.util.shape.view_as_windows()
        '''
        s0, s1 = arr.strides[:2]
        m1, n1 = arr.shape[:2]
        m2, n2 = sub_shape
        view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
        strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
        subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
        return subs

    def poolingOverlap(self, mat, ksize, stride=None, method='max', pad=False):
        '''Overlapping pooling on 2D or 3D data.

        <mat>: ndarray, input array to pool.
        <ksize>: tuple of 2, kernel size in (ky, kx).
        <stride>: tuple of 2 or None, stride of pooling window.
                  If None, same as <ksize> (non-overlapping pooling).
        <method>: str, 'max for max-pooling,
                       'mean' for mean-pooling.
        <pad>: bool, pad <mat> or not. If no pad, output has size
               (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
               if pad, output has size ceil(n/s).

        Return <result>: pooled matrix.
        '''

        m, n = mat.shape[:2]
        ky, kx = ksize
        if stride is None:
            stride = (ky, kx)
        sy, sx = stride

        _ceil = lambda x, y: int(np.ceil(x / float(y)))

        if pad:
            ny = _ceil(m, sy)
            nx = _ceil(n, sx)
            size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m, :n, ...] = mat
        else:
            mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

        view = self.asStride(mat_pad, ksize, stride)

        if method == 'max':
            result = np.nanmax(view, axis=(2, 3))
        else:
            result = np.nanmean(view, axis=(2, 3))

        return result

    def _get_overlay(self, cam, dm):
        _dm_depth = dm[:, :, 2:3] / np.max(dm[:, :, 2:3])
        _dm_cam_depth = dm[:, :, 3:4]
        _dm_intensity = dm[:, :, 4:5]
        _vis_pad_b = copy.deepcopy(_dm_cam_depth)

        p0 = np.percentile(_dm_cam_depth, 98) + 0.001

        _vis_pad_b[_dm_depth >= p0] = 0.

        _dm_ft_padding = np.concatenate([_vis_pad_b, _dm_cam_depth, _dm_intensity], -1) * 255.

        _dm_ft_padding_pool = self.poolingOverlap(_dm_ft_padding, ksize=[2, 2], stride=(2, 2), method='max', pad=False)
        cam_pool = self.poolingOverlap(cam, ksize=[2, 2], stride=(2, 2), method='max', pad=False)

        # _dm_ft_padding = cv2.convertScaleAbs(_dm_ft_padding.astype(np.uint8), alpha=1.4, beta=0)
        return cv2.addWeighted(_dm_ft_padding_pool.astype(np.uint8), 1.0, cam_pool.astype(np.uint8), 0.5, 0)

