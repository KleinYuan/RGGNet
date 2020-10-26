import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.lie_group as lie_group
import numpy as np
import liegroups


def SE3_to_se3(SE3_matrix):
    # This liegroups lib represent se3 with first 3 element as translation, which is different than us
    se3_rot_last = liegroups.SE3.log(liegroups.SE3.from_matrix(SE3_matrix))
    se3 = np.zeros_like(se3_rot_last)
    se3[:3] = se3_rot_last[3:]
    se3[3:] = se3_rot_last[:3]
    return se3

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


def get_compares_dict():
    compare_dict = {
        'calibnet': {
            'rpyxyz': np.array([0.0026426, 0.0157038, 0.0031621, 0.121, 0.03490, 0.07870]).astype(np.float32),
            'SE3': None,
            'se3_tf': None
        },
        'taylor_tro': {
            'rpyxyz': np.array([0.0076542, 0.0072328, 0.0027155, 0.063, 0.075, 0.020]).astype(np.float32),
            'SE3': None,
            'se3': None
        },
        'kitti': {
            'rpyxyz': np.array([0.174533, 0.174533, 0.174533, 0.100, 0.100, 0.100]).astype(np.float32),
            'SE3': None,
            'se3': None
        },
    }
    for _k, _v in compare_dict.items():
        _rpyxyz = _v['rpyxyz']
        _rm = rpy2rm(roll=_rpyxyz[0], pitch=_rpyxyz[1], yaw=_rpyxyz[2])
        _xyz = np.reshape(_rpyxyz[3:], (3, 1))
        _SE3 = np.concatenate([np.concatenate([_rm, _xyz], -1), np.reshape(np.array([0, 0, 0, 1]), (1, 4))], 0).astype(np.float32)
        _v['SE3'] = _SE3.astype(np.float32)
        _v['se3'] = SE3_to_se3(SE3_matrix=_SE3).astype(np.float32)
    print(compare_dict)
    return compare_dict


def test_se3():
    # rpy <--> xyz convention
    # http://web.mit.edu/2.05/www/Handout/HO2.PDF
    # examples:
    # [r, p, y, x, y, z]
    SE3_GROUP = SpecialEuclideanGroup(3, epsilon=np.finfo(np.float32).eps)
    metric = SE3_GROUP.left_canonical_metric

    identity = tf.constant(np.array([0., 0, 0, 0, 0, 0], dtype=np.float32), name='identity')
    r90 = tf.constant(np.array([1.5707963, 0., 0, 0, 0, 0], dtype=np.float32), name='r90')
    x10 =tf.constant(np.array([0., 0, 0, 10, 0, 0], dtype=np.float32), name='x10')
    r90_x10 = tf.constant(np.array([1.5707963, 0, 0, 10, 0, 0], dtype=np.float32), name='r90_x10')

    test_se3s = tf.constant(np.array([[1.5707963, 0, 0, 0, 0, 0], [0, 0, 0, 10, 0, 0], [1.5707963, 0, 0, 10, 0, 0]], dtype=np.float32), name='test_se3s')
    test_identities = tf.constant(np.array([[0., 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32),  name='test_se3s')

    dist_r90_to_identity = lie_group.loss(r90, identity, SE3_GROUP, metric)
    dist_x10_to_identity = lie_group.loss(x10, identity, SE3_GROUP, metric)
    dist_x10_to_r90 = lie_group.loss(x10, r90, SE3_GROUP, metric)
    psudo_riemannian_log_dist_x10_to_r90 = dist_r90_to_identity + dist_x10_to_identity  # Decoupled
    dist_r90_to_x10 = lie_group.loss(r90, x10, SE3_GROUP, metric)
    dist_r90_x10_to_identity = lie_group.loss(r90_x10, identity, SE3_GROUP, metric)

    dist_batch = lie_group.loss(test_se3s, test_identities, SE3_GROUP, metric)
    mean_dist_batch = tf.reduce_mean(dist_batch)

    test_dict = {
        'dist_r90_to_identity': dist_r90_to_identity,
        'dist_x10_to_identity': dist_x10_to_identity,
        'dist_x10_to_r90': dist_x10_to_r90,
        'psudo_riemannian_log_dist_x10_to_r90': psudo_riemannian_log_dist_x10_to_r90,
        'dist_r90_to_x10': dist_r90_to_x10,
        'dist_r90_x10_to_identity': dist_r90_x10_to_identity,
        'dist_batch': dist_batch,
        'mean_dist_batch': mean_dist_batch
    }

    _compare_dict = get_compares_dict()

    for _k, _v in _compare_dict.items():
        test_dict['dist_{}_to_identity'.format(_k)] = lie_group.loss(tf.constant(_v['se3'], name=_k), identity, SE3_GROUP, metric)

    with tf.Session() as sess:
        for _k, _v in test_dict.items():
            print("{} : {}".format(_k, sess.run(_v)))
    """
    dist_r90_to_identity : [[2.4674206]]
    dist_x10_to_identity : [[100.]]
    dist_x10_to_r90 : [[102.467415]]
    psudo_riemannian_log_dist_x10_to_r90 : [[102.46742]]
    dist_r90_to_x10 : [[102.46741]]
    dist_r90_x10_to_identity : [[102.46741]]
    dist_batch : [[  2.4674203], [100.       ], [102.46741  ]]
    mean_dist_batch : 68.31160736083984
    
    dist_calibnet_to_identity : [[0.02231666]]
    dist_taylor_tro_to_identity : [[0.01011221]]

    It can be seen that:
    - (None Euclidean) x10_to_r90 != x10_to_identity + r90_to_identity
    Several existing works that have benchmarked against the KITTI is :
    TAYLOR-TRO2016 (+/- 0.2 m, +/- 15 deg on drive 1030/0027): 0.01011221
    CalibNet-IROS2018 (+/- 0.2 m, +/- 10 deg): 0.02231666
    * Psudo logarithm mapping will have the following:
    dist_calibnet_se3_to_identity : [[0.02231641]]
    dist_taylor_se3_to_identity : [[0.01011236]]
    """


if __name__ == "__main__":
    test_se3()
