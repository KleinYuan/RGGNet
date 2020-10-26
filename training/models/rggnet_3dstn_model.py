import os
os.environ['GEOMSTATS_BACKEND'] = 'tensorflow'  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from ..core.base_model import BaseModel
from ..utils.tf_ops import tolerance_regularizer, batch_3dstn, batch_se3toSE3
from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.lie_group as lie_group
from ..utils.tf_rggnet import rggnet_forward


class Model(BaseModel):

    def init(self):
        self.model_name = self.config.name
        self.placeholders_configs = self.config.tensors.placeholders
        self.hyper_params = self.config.tensors.hyper_params
        self.batch_size = self.config.train.batch_size

    def define_net(self, training=True):
        with tf.name_scope(self.model_name):
            with tf.variable_scope('Placeholders'):
                self.x_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_dm.shape, name='x_dm')  # Depth Maps
                self.x_cam = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_cam.shape, name='x_cam')  # Camera
                self.x_R_rects = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_R_rects.shape, name='x_R_rects')  # Camera R Rects
                self.x_P_rects = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_P_rects.shape, name='x_P_rects')  # Camera R Rects
                self.y_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.outputs.y_dm.shape, name='y_dm')  # Target Depth Maps
                self.y_se3param = tf.placeholder(tf.float32, [None] + self.placeholders_configs.outputs.y_se3param.shape, name='y_se3param')  # Target se3param
                self.is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.name_scope("pre_process"):
                self.x_cam_augmented = tf.image.random_brightness(self.x_cam, max_delta=0.5)
                self.x_cam_train = tf.cond(self.is_training, lambda: self.x_cam_augmented, lambda: self.x_cam)

            self.x_dm_ft = tf.identity(self.x_dm[:, :, :, 3:5], name='x_dm_ft')
            self.y_hat_se3param, self.x_dm_ft_pool, self.x_cam_pool = rggnet_forward(x_dm_ft=self.x_dm_ft, x_cam=self.x_cam_train, is_training=self.is_training)

        _, H, W, C = self.x_dm.get_shape().as_list()

        with tf.name_scope('Reprojection'):
            with tf.name_scope('se3_to_SE3'):
                _SE3 = batch_se3toSE3(self.y_hat_se3param)
            with tf.name_scope('Conditional_3D_STN'):
                self.y_hat_dm = batch_3dstn(
                    tfs=_SE3, #[B, 3, 4], tfs: transformation matrix s
                    x_dm=self.x_dm, # only keep the cam-z dimension
                    R_rects=self.x_R_rects,
                    P_rects=self.x_P_rects,
                    H=H,
                    W=W
                )

    def define_loss(self):
        with tf.name_scope('Loss'):
            print("Loss ===============================================Start")
            with tf.variable_scope('Resize_for_VAE'):
                y_hat_dm_resize = tf.image.resize_bilinear(
                    self.y_hat_dm[:, :, :, 3:5],
                    size=(self.hyper_params.vae_h, self.hyper_params.vae_w),
                    align_corners=False,
                    half_pixel_centers=False,
                    name='y_hat_dm_resize'
                )
                y_cam_resize = tf.image.resize_bilinear(
                    self.x_cam_train,
                    size=(self.hyper_params.vae_h, self.hyper_params.vae_w),
                    align_corners=False,
                    half_pixel_centers=False,
                    name='y_cam_resize'
                )
                vae_inputs = tf.concat([y_cam_resize, y_hat_dm_resize], -1, name='vae_input')
                print("    [vae inputs]           {}".format(vae_inputs))

            self.tolerance_regularization, _ = tolerance_regularizer(inputs=vae_inputs,
                                                                     vae_latent_dim=self.hyper_params.vae_latent_dim,
                                                                     is_training=self.is_training)
            self.frozen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Tolerance_Regularization')

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
            self.loss = tf.identity(self.loss_geodesic + self.hyper_params.regularizer_factor * self.tolerance_regularization, name='loss')

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
            tf.summary.scalar('tolerance_regularization', self.tolerance_regularization),
            tf.summary.scalar('SE3_Loss/tamas_iccv2013', tf.constant(0.00478775, name='tamas_iccv2013_SE3_loss')),
            tf.summary.scalar('SE3_Loss/taylor_tro2016', tf.constant(0.01011221, name='taylor_tro2016_SE3_loss')),
            tf.summary.scalar('SE3_Loss/calibnet', tf.constant(0.02231666, name='calibnet_SE3_loss'))
        ]

    def define_tensor_dict(self):
        self.tensor_dict = {
            'x_dm': self.x_dm,
            'x_cam': self.x_cam,
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

    def define_optimizer(self):
        print('[Cosine Restart] Defining Optimizer ...')
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
            print('[Optimizing Var List] {}'.format(_var_list))
        else:
            _var_list = []
            for _var in tf.trainable_variables():
                if _var in self.frozen_vars:
                    print("Skip: {}".format(_var))
                else:
                    _var_list.append(_var)
        if len(_var_list) == 0:
            print('[Optimizing Var List] Optimizing all variables')
            _var_list = None
        optimizer = optimizer.minimize(self.loss, var_list=_var_list, global_step=global_step)
        self.optimizer = tf.group([optimizer, update_ops])
