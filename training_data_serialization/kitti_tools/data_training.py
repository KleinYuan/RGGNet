from kitti_tools.data_templates import TrainingData
import numpy as np
import copy
from numpy.linalg import inv
import cv2
from kitti_tools.transform import generate_depth_map, update_depth_map, SE3_to_se3, rpy2rm
from kitti_tools.reduce_lidar_line import reduce_lidar_line


def sample_noise(range_rot, range_trans):
    random_r = np.random.uniform(low=-range_rot, high=range_rot, size=(1, 1))
    random_p = np.random.uniform(low=-range_rot, high=range_rot, size=(1, 1))
    random_y = np.random.uniform(low=-range_rot, high=range_rot, size=(1, 1))
    random_tx = np.random.uniform(low=-range_trans, high=range_trans, size=(1, 1))
    random_ty = np.random.uniform(low=-range_trans, high=range_trans, size=(1, 1))
    random_tz = np.random.uniform(low=-range_trans, high=range_trans, size=(1, 1))

    return random_r, random_p, random_y, random_tx, random_ty, random_tz


def random_transform(raw_transform, range_rot_rad, range_trans_m):
    """
    Inject random rotation/translation to input transform.
    So that we obtain a mis-calibrated transform and ground truth transform
    which if you add ground truth transform to mis-calibrated transform, you will get
    original transform

    :param transform: (3, 2) matrix with (:, 0) as rotation and (:, 1) as translation
    :return:
        - init_transformmed
        - gt_transform

    init_transformmed + gt_transform = transform
    """
    sys_transform = copy.deepcopy(raw_transform)
    sys_transform_rot_rm = sys_transform[:, :3]
    sys_transform_trans = sys_transform[:, [-1]]

    random_r, random_p, random_y, random_tx, random_ty, random_tz = sample_noise(range_rot_rad, range_trans_m)
    random_rotation = np.concatenate([random_r, random_p, random_y], axis=0).reshape([3, 1])
    random_translation = np.concatenate([random_tx, random_ty, random_tz], axis=0).reshape([3, 1])

    noise_rm = rpy2rm(
        roll=random_rotation[0][0],
        pitch=random_rotation[1][0],
        yaw=random_rotation[2][0])

    init_transform_rot_rm = np.dot(inv(noise_rm), sys_transform_rot_rm)
    target_transform = np.concatenate([noise_rm, random_translation], axis=-1)
    init_transform = np.concatenate([init_transform_rot_rm, sys_transform_trans - random_translation], axis=-1)
    # Rotation Matrix 3 x 3 --> [3, 3]
    # Translation    3 x 1 --> [3, 1]
    # Concate       [3 , 4]

    return init_transform, target_transform


def generate_dm(pts, tx, l_rect_matrix, l_proj_matrix, r_rect_matrix, r_proj_matrix, img_size, rot_range, trans_range):
    H, W = img_size
    init_tx, target_tx = random_transform(tx, rot_range, trans_range)
    l_x_dm = generate_depth_map(
        pts_3d_in=pts,
        transform_rm_3by4=init_tx,
        R_rect=l_rect_matrix,
        P_rect=l_proj_matrix,
        H=H,
        W=W)

    l_y_dm = update_depth_map(
        depth_map=l_x_dm,
        transform_rm_3by4=target_tx,
        R_rect=l_rect_matrix,
        P_rect=l_proj_matrix,
        H=H,
        W=W)

    r_x_dm = generate_depth_map(
        pts_3d_in=pts,
        transform_rm_3by4=init_tx,
        R_rect=r_rect_matrix,
        P_rect=r_proj_matrix,
        H=H,
        W=W)

    r_y_dm = update_depth_map(
        depth_map=r_x_dm,
        transform_rm_3by4=target_tx,
        R_rect=r_rect_matrix,
        P_rect=r_proj_matrix,
        H=H,
        W=W)

    return l_x_dm, l_y_dm, r_x_dm, r_y_dm, target_tx

def generate_single_example(raw_data, range_rot_deg, range_trans_m, vae_resized_H, vae_resized_W, force_shape=None, reduce_lidar_line_to=None):
    """
    raw_data: RawData Obj
    range_rot_deg: off calibration range in rotation  in degree
    range_trans: off calibration range in translation in meter

    def _set_Xs(self):
        # For VAE
        self.x_dm_ft_resized = None
        self.x_cam_resized = None

        # For RGGNet
        self.x_dm = None
        self.x_cam = None
        self.x_R_rects = None
        self.x_P_rects = None

    def _set_Ys(self):
        self.y_se3param = None
        self.y_dm = None
    """

    training_data = TrainingData(raw_data=raw_data)
    range_rot_rad = float(range_rot_deg) * np.pi / 180.
    if reduce_lidar_line_to is not None:
        training_data.pts_data = reduce_lidar_line(training_data.pts_data, reduce_lidar_line_to=reduce_lidar_line_to)

    # Assign whatever available
    training_data.x_cam = training_data.rgb_data
    training_data.r_x_cam = training_data.r_rgb_data
    # TODO: This could not be the best idea --> Only happened to cam with slightly (1 pixel) different examples
    if force_shape is not None:
        training_data.x_cam = cv2.resize(training_data.x_cam, (force_shape[1], force_shape[0]), interpolation=cv2.INTER_LINEAR)

    training_data.x_R_rects = training_data.raw_data.cam_mat_R
    training_data.x_P_rects = training_data.raw_data.cam_mat_P
    training_data.r_x_R_rects = training_data.raw_data.r_cam_mat_R
    training_data.r_x_P_rects = training_data.raw_data.r_cam_mat_P
    training_data.x_cam_resized = cv2.resize(training_data.rgb_data, (vae_resized_W, vae_resized_H), interpolation=cv2.INTER_LINEAR)

    # Compute x_dm, y_dm, x_dm_ft_resized, y_se3param
    # H, W = training_data.x_cam.shape[:2]
    # _init_transform_rm_3by4, _target_transform_rm_3by4 = random_transform(training_data.raw_data.transform_mat, range_rot_rad, range_trans_m)

    # training_data.x_dm = generate_depth_map(
    #         pts_3d_in=training_data.pts_data,
    #         transform_rm_3by4=_init_transform_rm_3by4,
    #         R_rect=training_data.x_R_rects,
    #         P_rect=training_data.x_P_rects,
    #         H=H,
    #         W=W)

    # training_data.y_dm = update_depth_map(
    #     depth_map=training_data.x_dm,
    #     transform_rm_3by4=_target_transform_rm_3by4,
    #     R_rect=training_data.x_R_rects,
    #     P_rect=training_data.x_P_rects,
    #     H=H,
    #     W=W)

    img_size = training_data.x_cam.shape[:2]
    l_x_dm, l_y_dm, r_x_dm, r_y_dm, target_tx = generate_dm(
        training_data.pts_data,
        training_data.raw_data.transform_mat,
        training_data.x_R_rects,
        training_data.x_P_rects,
        training_data.r_x_R_rects,
        training_data.r_x_P_rects,
        img_size,
        range_rot_rad,
        range_trans_m
    )
    training_data.x_dm, training_data.y_dm = l_x_dm, l_y_dm
    training_data.r_x_dm, training_data.r_y_dm = r_x_dm, r_y_dm

    training_data.x_dm_ft_resized = cv2.resize(training_data.y_dm[:, :, 3:5], (vae_resized_W, vae_resized_H), interpolation=cv2.INTER_LINEAR)

    # _target_SE3 = np.concatenate([_target_transform_rm_3by4, np.reshape(np.array([0. , 0., 0., 1.]), newshape=(1, 4))], 0).astype(np.float32)
    _target_SE3 = np.concatenate([target_tx, np.reshape(np.array([0. , 0., 0., 1.]), newshape=(1, 4))], 0).astype(np.float32)
    training_data.y_se3param = SE3_to_se3(SE3_matrix=_target_SE3).astype(np.float32)

    return training_data
