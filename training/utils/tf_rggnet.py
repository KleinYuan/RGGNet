import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v2


def rggnet_forward(x_dm_ft, x_cam, is_training):
    print("Forward ===============================================Start")
    slim = tf.contrib.slim
    with tf.variable_scope('Pre_Process_Pool'):
        x_dm_ft_pool = tf.nn.avg_pool(x_dm_ft, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_cam_pool = tf.nn.avg_pool(x_cam, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        print("    [Pre_Process_Pool] {} --> {}".format(x_dm_ft, x_dm_ft_pool))
        print("    [Pre_Process_Pool] {} --> {}".format(x_cam, x_cam_pool))
    with tf.name_scope('Forward'):
        with tf.name_scope('Feature_Extraction'):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                f_dm, endpoint_dm = resnet_v2.resnet_v2_50(x_dm_ft_pool, None, is_training=is_training,
                                                           scope='depth_map')
                print("    [Feature DM ]:                              {}".format(f_dm))
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                f_cam, endpoint_cam = resnet_v2.resnet_v2_50(x_cam_pool, None, is_training=is_training)
                print("    [Feature CAM]:                              {}".format(f_cam))
        with tf.variable_scope('Fuse'):
            feature = tf.concat([f_dm, f_cam], -1)
            print("    [Feature Concat]:                            {}".format(f_cam))
            flattened_feature = tf.layers.flatten(feature)
        print("    [Flattened ]:                             {}".format(flattened_feature))
        with tf.variable_scope('MLPs'):
            mlp_1 = tf.layers.dense(flattened_feature, 2048, activation=tf.nn.leaky_relu, name='mlp1')
            mlp_1 = tf.layers.dropout(mlp_1, training=is_training)
            mlp_2 = tf.layers.dense(mlp_1, 1024, activation=tf.nn.leaky_relu, name='mlp2')
            mlp_2 = tf.layers.dropout(mlp_2, training=is_training)
            mlp_3 = tf.layers.dense(mlp_2, 512, activation=tf.nn.leaky_relu, name='mlp3')
            mlp_3 = tf.layers.dropout(mlp_3, training=is_training)
            print("    [MLPs ]:                              {}".format(mlp_3))
        with tf.variable_scope('Regressor'):
            with tf.variable_scope('rotation'):
                # This is the angle-axis representation, with a range of [-20, +20] deg
                out_rot = tf.layers.dense(mlp_3, 3, activation=tf.nn.tanh, name='rot')
                out_rot = out_rot * np.pi * (20. / 180.)
            with tf.variable_scope('translate'):
                out_trans = tf.layers.dense(mlp_3, 3, activation=None, name='trans')

            y_hat_se3param = tf.concat([out_rot, out_trans], -1, name='y_hat_se3param')
            print("    [y_hat_se3param]:                              {}".format(y_hat_se3param))
    print("Forward ===============================================End")
    return y_hat_se3param, x_dm_ft_pool, x_cam_pool
