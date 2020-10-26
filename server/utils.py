import os
import yaml
import numpy as np
from box import Box
import geomstats.lie_group as lie_group
from geomstats.special_euclidean_group import SpecialEuclideanGroup
import liegroups

SE3_GROUP = SpecialEuclideanGroup(3, epsilon=np.finfo(np.float32).eps)
metric = SE3_GROUP.left_canonical_metric


def load_config(config_fp):
    config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
    config = Box(yaml.load(open(config_abs_fp, 'r').read()))
    return config


def cal_se3_error(se3_pred, se3_true):
    se3_error = lie_group.loss(se3_pred, se3_true, SE3_GROUP, metric)
    return se3_error


def SE3_to_se3(SE3_matrix):
    # This liegroups lib represent se3 with first 3 element as translation, which is different than us
    se3_rot_last = liegroups.SE3.log(liegroups.SE3.from_matrix(SE3_matrix, normalize=True))
    se3 = np.zeros_like(se3_rot_last)
    se3[:3] = se3_rot_last[3:]
    se3[3:] = se3_rot_last[:3]
    return se3


def se3_to_SE3(se3):
    SE3 = liegroups.SE3.exp([se3[3], se3[4], se3[5], se3[0], se3[1], se3[2]]).as_matrix()
    return SE3
