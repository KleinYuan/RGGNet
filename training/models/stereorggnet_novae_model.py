import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from ..models.rggnet_model import Model as BaseModel
from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.lie_group as lie_group

from tensorflow.contrib.slim.nets import resnet_v2

def cam_dm_forward(x_dm_ft, x_cam, is_training, prefix):
    slim = tf.contrib.slim
    with tf.variable_scope(f'{prefix}_Pre_Process_Pool'):
        x_dm_ft_pool = tf.nn.avg_pool(x_dm_ft, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        x_cam_pool = tf.nn.avg_pool(x_cam, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print("[Pre_Process_Pool] {} --> {}".format(x_dm_ft, x_dm_ft_pool))
    print("[Pre_Process_Pool] {} --> {}".format(x_cam, x_cam_pool))

    with tf.name_scope(f'{prefix}'):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            f_cam, _ = resnet_v2.resnet_v2_50(x_cam_pool, None, is_training=is_training, scope=f'{prefix.lower()}_cam')
            print("\t[Feature CAM]:\t{}".format(f_cam))
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            f_dm, _ = resnet_v2.resnet_v2_50(x_dm_ft_pool, None, is_training=is_training, scope=f'{prefix.lower()}_dm')
            print("\t[Feature DM]:\t{}".format(f_dm))

    with tf.variable_scope(f'{prefix}_Fuse'):
        feature = tf.concat([f_dm, f_cam], -1)
        print("\t[Feature Concat]:\t{}".format(f_cam))
        flattened_feature = tf.layers.flatten(feature)
        print("\t[Flattened ]:\t{}".format(flattened_feature))
    with tf.variable_scope(f'{prefix}_MLPs'):
        mlp_1 = tf.layers.dense(flattened_feature, 2048, activation=tf.nn.leaky_relu, name='mlp1')
        mlp_1 = tf.layers.dropout(mlp_1, training=is_training)
        mlp_2 = tf.layers.dense(mlp_1, 1024, activation=tf.nn.leaky_relu, name='mlp2')
        mlp_2 = tf.layers.dropout(mlp_2, training=is_training)
        mlp_3 = tf.layers.dense(mlp_2, 512, activation=tf.nn.leaky_relu, name='mlp3')
        mlp_3 = tf.layers.dropout(mlp_3, training=is_training)
        print("\t[{} MLP3]:\t{}".format(prefix, mlp_3))

    return mlp_3


def stereo_rggnet_forward(x_dm_ft, x_cam, r_x_dm_ft, r_x_cam, is_training):
    print("Forward STEREO ===============================================Start")
    l_mlp3 = cam_dm_forward(x_dm_ft, x_cam, is_training, 'left')
    r_mlp3 = cam_dm_forward(r_x_dm_ft, r_x_cam, is_training, 'right')
    merged_mlp = tf.concat([l_mlp3, r_mlp3], -1)

    mlp_4 = tf.layers.dense(merged_mlp, 512, activation=tf.nn.leaky_relu, name='mlp4')
    mlp_4 = tf.layers.dropout(mlp_4, training=is_training)

    with tf.variable_scope('Regressor'):
        with tf.variable_scope('rotation'):
            # This is the angle-axis representation, with a range of [-20, +20] deg
            out_rot = tf.layers.dense(mlp_4, 3, activation=tf.nn.tanh, name='rot')
            out_rot = out_rot * np.pi * (20. / 180.)
        with tf.variable_scope('translate'):
            out_trans = tf.layers.dense(mlp_4, 3, activation=None, name='trans')

    y_hat_se3param = tf.concat([out_rot, out_trans], -1, name='y_hat_se3param')
    print("    [y_hat_se3param]:                              {}".format(y_hat_se3param))
    print("Forward ===============================================End")
    return y_hat_se3param

class Model(BaseModel):
    def define_tensor_dict(self):
        self.tensor_dict = {
            'x_dm': self.x_dm,
            'x_cam': self.x_cam,
            'r_x_dm': self.r_x_dm,
            'r_x_cam': self.r_x_cam,
            'x_R_rects': self.x_R_rects,
            'x_P_rects': self.x_P_rects,
            'y_dm': self.y_dm,
            'y_se3param': self.y_se3param,
            'is_training':
                {
                    'tensor': self.is_training,
                    'value': {
                        'train': True,
                        'inference': False
                    }
                }
        }

    def define_net(self, training=True):
        with tf.name_scope(self.model_name):
            with tf.variable_scope('Placeholders'):
                self.x_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_dm.shape, name='x_dm')  # Depth Maps
                self.x_cam = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_cam.shape, name='x_cam')  # Camera

                self.r_x_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.r_x_dm.shape, name='r_x_dm')  # Depth Maps
                self.r_x_cam = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.r_x_cam.shape, name='r_x_cam')  # Camera

                self.x_R_rects = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_R_rects.shape, name='x_R_rects')  # Camera R Rects
                self.x_P_rects = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_P_rects.shape, name='x_P_rects')  # Camera R Rects

                self.y_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.outputs.y_dm.shape, name='y_dm')  # Target Depth Maps

                self.y_se3param = tf.placeholder(tf.float32, [None] + self.placeholders_configs.outputs.y_se3param.shape, name='y_se3param')  # Target se3param
                self.is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.name_scope("pre_process"):
                self.x_cam_augmented = tf.image.random_brightness(self.x_cam, max_delta=0.5)
                self.x_cam_train = tf.cond(self.is_training, lambda: self.x_cam_augmented, lambda: self.x_cam)
                self.r_x_cam_augmented = tf.image.random_brightness(self.r_x_cam, max_delta=0.5)
                self.r_x_cam_train = tf.cond(self.is_training, lambda: self.r_x_cam_augmented, lambda: self.r_x_cam)

            self.x_dm_ft = tf.identity(self.x_dm[:, :, :, 3:5], name='x_dm_ft')
            self.r_x_dm_ft = tf.identity(self.r_x_dm[:, :, :, 3:5], name='r_x_dm_ft')
            self.y_hat_se3param = stereo_rggnet_forward(x_dm_ft=self.x_dm_ft, r_x_dm_ft=self.r_x_dm_ft,
                                                        x_cam=self.x_cam_train, r_x_cam=self.r_x_cam_train,
                                                        is_training=self.is_training)

    def define_loss(self):
        with tf.name_scope('Loss'):
            print("Loss ===============================================Start")
            with tf.variable_scope('SE3_Loss'):
                print("    [SE3 Loss]           ##################################")
                # TODO: Hard coded batch number!!! BAD!!!
                se3_pred = tf.reshape(self.y_hat_se3param, (self.batch_size, 6))
                se3_true = tf.reshape(self.y_se3param, (self.batch_size, 6))
                print("    [y_hat_se3param]           {}".format(self.y_hat_se3param))
                print("    [y_se3param    ]           {}".format(self.y_se3param))
                print("    [se3_pred     ]           {}".format(se3_pred))
                print("    [se3_true     ]           {}".format(se3_true))
                SE3_GROUP = SpecialEuclideanGroup(3, epsilon=np.finfo(np.float32).eps)
                metric = SE3_GROUP.left_canonical_metric
                self.loss_geodesic = tf.reduce_mean(lie_group.loss(se3_pred, se3_true, SE3_GROUP, metric))

        with tf.name_scope('Loss'):
            self.loss = tf.identity(self.loss_geodesic, name='loss')

    def define_summary_list(self):
        """
            Several existing works that have benchmarked against the KITTI is :
                TAMAS-ICCV2013 (+/- 0.2 m, +/- 15 deg on drive 0005): 0.00478775
                TAYLOR-TRO2016 (+/- 0.2 m, +/- 15 deg on drive 0027): 0.01011221
                CalibNet-IROS2018 (+/- 0.2 m, +/- 10 deg): 0.02231666
        """
        self.summary_list = [
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar('SE3_Loss/RggNet', self.loss_geodesic),
            tf.summary.scalar('SE3_Loss/tamas_iccv2013', tf.constant(0.00478775, name='tamas_iccv2013_SE3_loss')),
            tf.summary.scalar('SE3_Loss/taylor_tro2016', tf.constant(0.01011221, name='taylor_tro2016_SE3_loss')),
            tf.summary.scalar('SE3_Loss/calibnet', tf.constant(0.02231666, name='calibnet_SE3_loss'))
        ]

    def define_optimizer(self):
        self.logger.info('[Cosine Restart] Defining Optimizer ...')
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.cosine_decay_restarts(
            learning_rate=self.start_lr,
            global_step=global_step,
            first_decay_steps=self.decay_steps,
            m_mul=self.decay_rate
        )
        tf.summary.scalar("lr", self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.config.train.continue_training:
            _var_list = [v for v in tf.trainable_variables() if
                         v.name.split('/')[0] in self.config.train.optimizer_var_list]
            self.logger.info('[Optimizing Var List] {}'.format(_var_list))
        else:
            _var_list = []
        if len(_var_list) == 0:
            self.logger.info('[Optimizing Var List] Optimizing all variables')
            _var_list = None
        optimizer = optimizer.minimize(self.loss, var_list=_var_list, global_step=global_step)
        self.optimizer = tf.group([optimizer, update_ops])
