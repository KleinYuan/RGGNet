import tensorflow as tf
from ..utils.tf_ops import tolerance_regularizer
from ..core.base_model import BaseModel


class Model(BaseModel):

    def init(self):
        self.model_name = self.config.name
        self.placeholders_configs = self.config.tensors.placeholders
        self.hyper_params = self.config.tensors.hyper_params

    def define_visualization(self):
        with tf.variable_scope("visualization"):
            with tf.variable_scope("inputs"):
                _x_cam_resized_255 = self.x_cam_resized[0:1, :, :, :] * 255.
                tf.summary.image(
                    name='camera',
                    tensor=_x_cam_resized_255,
                    max_outputs=10,
                    collections=None,
                    family=None
                )

                _x_dm_ft_resized_255 = self.x_dm_ft_resized[0:1, :, :, :] * 255.
                tf.summary.image(
                    name='depth_map_intensity',
                    tensor=_x_dm_ft_resized_255[0:1, :, :, 0:1],
                    max_outputs=10,
                    collections=None,
                    family=None
                )
                tf.summary.image(
                    name='depth_map_z_cam_norm',
                    tensor=_x_dm_ft_resized_255[0:1, :, :, 1:2],
                    max_outputs=10,
                    collections=None,
                    family=None
                )

                _x_cam_resized_augmented_255 = self.x_cam_resized_augmented[0:1, :, :, :] * 255.
                tf.summary.image(
                    name='camera_augmented',
                    tensor=_x_cam_resized_augmented_255,
                    max_outputs=10,
                    collections=None,
                    family=None
                )
            with tf.variable_scope("outputs"):
                _x_hat_cam_255 = self.x_hat_pm[0:1, :, :, 0:3] * 255.
                tf.summary.image(
                    name='camera',
                    tensor=_x_hat_cam_255,
                    max_outputs=10,
                    collections=None,
                    family=None
                )
                _x_hat_dm_255 = self.x_hat_pm[0:1, :, :, :] * 255.
                tf.summary.image(
                    name='depth_map_intensity',
                    tensor=_x_hat_dm_255[0:1, :, :, 0:1],
                    max_outputs=10,
                    collections=None,
                    family=None
                )
                tf.summary.image(
                    name='depth_map_z_cam_norm',
                    tensor=_x_hat_dm_255[0:1, :, :, 1:2],
                    max_outputs=10,
                    collections=None,
                    family=None
                )

    def define_net(self, training=True):
        with tf.name_scope(self.model_name):
            with tf.variable_scope("vae_train_inputs"):
                # This variable scopes are to be removed
                self.x_dm_ft_resized = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_dm_ft_resized.shape,
                                           name='x_dm_ft_resized')  # Depth Maps Resized
                self.x_cam_resized = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_cam_resized.shape,
                                            name='x_cam_resized')  # Camera Resized

                self.is_training = tf.placeholder(tf.bool, name='is_training')
                # Data Augmentation!!!
                self.x_cam_resized_augmented = tf.image.random_brightness(self.x_cam_resized, max_delta=0.5)
                _x_cam_resized_train = tf.cond(self.is_training, lambda: self.x_cam_resized_augmented, lambda: self.x_cam_resized)
                vae_inputs = tf.concat([_x_cam_resized_train, self.x_dm_ft_resized], -1, name='vae_input')
                print("    [vae inputs]           {}".format(vae_inputs))

            # Below are to be reused.
            self.ELBO, self.x_hat_pm = tolerance_regularizer(
                inputs=vae_inputs,
                vae_latent_dim=self.hyper_params.vae_latent_dim,
                is_training=self.is_training)

            self.define_visualization()

    def define_tensor_dict(self):
        self.tensor_dict = {
            'x_dm_ft_resized': self.x_dm_ft_resized,
            'x_cam_resized': self.x_cam_resized,
            'is_training':
                {
                    'tensor': self.is_training,
                    'value': {
                        'train': True,
                        'inference': False
                    }
                }
        }

    def define_loss(self):
        self.logger.info('Defining loss ...')
        with tf.variable_scope('loss'):
            self.loss = tf.identity(self.ELBO, name='ELBO')

    def define_summary_list(self):
        self.summary_list = [
            tf.summary.scalar("vae_loss", self.loss)
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
