import numpy as np
from copy import deepcopy
import liegroups
import math


def rm2rpy(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # x: roll
    # y: pitch
    # z: yaw
    return np.array([x, y, z])

# http://web.mit.edu/2.05/www/Handout/HO2.PDF
def rpy2rm(roll, pitch, yaw):
    rotation_matrix = np.array(
        [
            [np.cos(yaw) * np.cos(pitch),
             np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
             np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
             ],
            [np.sin(yaw) * np.cos(pitch),
             np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
             np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
             ],
            [-np.sin(pitch),
             np.cos(pitch) * np.sin(roll),
             np.cos(pitch) * np.cos(roll)
             ]
        ])
    return rotation_matrix


def SE3_to_se3(SE3_matrix):
    # This liegroups lib represent se3 with first 3 element as translation, which is different than us
    se3_rot_last = liegroups.SE3.log(liegroups.SE3.from_matrix(SE3_matrix, normalize=True))
    se3 = np.zeros_like(se3_rot_last)
    se3[:3] = se3_rot_last[3:]
    se3[3:] = se3_rot_last[:3]
    return se3


def cart2hom(pts_3d):
    """
    Input: nx3 points in Cartesian
    Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_pts_to_image(pts_3d, transform_rm_3by4, R_rect, P_rect, verbose=True):
    """
    - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
    - P_rect_xx: 3x4 projection matrix after rectification
    :param pts_3d: (?, 3) numpy array
    :param R_rect: (3, 3) numpy array
    :param P_rect: (3, 4) numpy array
    :param transform_rm_3by4: (3, 4) numpy array
    :return:
    """
    _transform_rm_3by4 = deepcopy(transform_rm_3by4)
    _R_rect = deepcopy(R_rect)
    _P_rect = deepcopy(P_rect)
    assert transform_rm_3by4.shape == (3, 4), "Transform RM XYZ shape is not correct!"
    if verbose:
        print("----------------PROJECT Points To Image--------------------------")
        print("Input Point Number: {}".format(pts_3d.shape[0]))
        print("Input Point (Cartesian) Example: {}".format(pts_3d[0]))
    pts_3d_velo = cart2hom(pts_3d)
    if verbose:
        print("Homogeneous Point Example: {}".format(pts_3d_velo[0]))
    pts_3d_ref = np.dot(pts_3d_velo, np.transpose(_transform_rm_3by4))
    # viz_combined_points(pts_3d_ref[:, :3], pts_3d[:, :3])
    # pts_3d_rect = np.transpose(np.dot(R_rect, np.transpose(pts_3d_ref)))
    pts_3d_rect = np.dot(pts_3d_ref, np.transpose(_R_rect))
    if verbose:
        print("pts_3d_rect: {}".format(pts_3d_rect.shape))
    pts_3d_rect = cart2hom(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(_P_rect))
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_on_img = pts_2d[:, 0:3]
    return pts_on_img, pts_3d_ref


def generate_depth_map(pts_3d_in, transform_rm_3by4, R_rect, P_rect, H, W, verbose=False):
    # Project Points to Image, still points but x, y ,z --> u, v
    # Each points include (x, y, z, intensity)
    # When do projection, we only need (x, y, z)
    pts_3d = deepcopy(pts_3d_in)
    clip_distance = 0
    transform_rm_3by4_copy = deepcopy(transform_rm_3by4)
    pts_cam_without_intensity, pts_rec_without_intensity = project_pts_to_image(
        pts_3d=pts_3d[:, :3],
        transform_rm_3by4=transform_rm_3by4_copy,
        R_rect=R_rect,
        P_rect=P_rect,
        verbose=verbose
    )
    # Get back intensity channel
    pts_cam = np.concatenate((pts_cam_without_intensity, pts_3d[:, [-1]]), axis=-1)

    # Filter Points not within range
    condition = (pts_cam[:, 0] < W) & \
                (pts_cam[:, 0] >= 0) & \
                (pts_cam[:, 1] < H) & \
                (pts_cam[:, 1] >= 0) & \
                (pts_3d[:, 0] > clip_distance)

    pts_cam_fov = pts_cam[condition]
    pts_rec_fov = pts_rec_without_intensity[condition]

    x, y, z, i = pts_cam_fov[:, 0], pts_cam_fov[:, 1], pts_cam_fov[:, 2], pts_cam_fov[:, 3]
    _x, _y, _z = pts_rec_fov[:, 0], pts_rec_fov[:, 1], pts_rec_fov[:, 2],

    phi_ = x.astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta_ = y.astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = np.zeros((H, W, 5))

    depth_map[theta_, phi_, 0] = _x
    depth_map[theta_, phi_, 1] = _y
    depth_map[theta_, phi_, 2] = _z
    try:
        zcam_normed = deepcopy(z) / np.max(z)
    except:
        zcam_normed = np.zeros_like(z)
    depth_map[theta_, phi_, 3] = i  # (0~1)
    depth_map[theta_, phi_, 4] = zcam_normed  # (0~1)

    return depth_map


def update_depth_map(depth_map, transform_rm_3by4, R_rect, P_rect, H, W):
    """
    depths maps are points on cam frame but data encoded with of rec frame
    """
    depth_map_in = deepcopy(depth_map)
    transform_rm_3by4_copy = deepcopy(transform_rm_3by4)
    pad = np.ones([H, W, 1])

    # Step1: Normalize back
    depth_map_xyz = depth_map_in[:, :, :3]
    # Step2: Transform
    depth_map_xyz_hom = np.concatenate([depth_map_xyz, pad], -1)
    depth_map_xyz_transformed = np.dot(depth_map_xyz_hom, np.transpose(transform_rm_3by4_copy))

    # Step2: Compute data on cam frame
    depth_map_xyz_transformed_rect = np.dot(depth_map_xyz_transformed, np.transpose(R_rect))
    depth_map_xyz_transformed_rect_hom = np.concatenate([depth_map_xyz_transformed_rect, pad], -1)

    depth_map_xyz_cam = np.dot(depth_map_xyz_transformed_rect_hom, np.transpose(P_rect))
    depth_map_xyz_cam[:, :, 0] /= depth_map_xyz_cam[:, :, 2]
    depth_map_xyz_cam[:, :, 1] /= depth_map_xyz_cam[:, :, 2]

    # Step3: Add back intensity
    # depth_map_transformed = np.concatenate([depth_map_xyz_transformed, depth_map_in[:, :, [-1]]], -1)

    x, y, z = depth_map_xyz_cam[:, :, 0], depth_map_xyz_cam[:, :, 1], depth_map_xyz_cam[:, :, 2]
    _x, _y, _z = depth_map_xyz_transformed[:, :, 0], depth_map_xyz_transformed[:, :, 1], depth_map_xyz_transformed[:, :, 2]
    i = depth_map_in[:, :, -2]

    depth_map_x_cam = x.astype(int)
    depth_map_x_cam = np.clip(depth_map_x_cam, 0, W-1)

    depth_map_y_cam = y.astype(int)
    depth_map_y_cam = np.clip(depth_map_y_cam, 0, H-1)

    depth_map_next = np.zeros((H, W, 5), np.float32)
    depth_map_next[depth_map_y_cam, depth_map_x_cam, 0] = _x
    depth_map_next[depth_map_y_cam, depth_map_x_cam, 1] = _y
    depth_map_next[depth_map_y_cam, depth_map_x_cam, 2] = _z

    try:
        zcam_normed = deepcopy(z) / np.max(z)
    except:
        zcam_normed = np.zeros_like(z)
    depth_map_next[depth_map_y_cam, depth_map_x_cam, 3] = i
    depth_map_next[depth_map_y_cam, depth_map_x_cam, 4] = zcam_normed

    return depth_map_next
