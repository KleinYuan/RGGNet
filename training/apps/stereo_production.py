import os
import yaml
from box import Box
import tensorflow as tf
from ..utils.inference import create_inference_graph
from ..utils.freeze import freeze_graph_from_file
from ..models.stereorggnet_novae_model import stereo_rggnet_forward
import fire


class InferenceModel(object):
    """
    This is a modified graph based on training graph, usually:
            - Remove redundant operators
            - Replace placeholder of tf.bool with constant
            - Replace placeholder of is_training False
    """
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_name = self.config.name
        self.placeholders_configs = self.config.tensors.placeholders
        self.hyper_params = self.config.tensors.hyper_params
        self.batch_size = self.config.train.batch_size
        self.define_graph()

    def define_net(self):
        with tf.name_scope(self.model_name):
            with tf.variable_scope('Placeholders'):
                self.x_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_dm.shape, name='x_dm')  # Depth Maps
                self.x_cam = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_cam.shape, name='x_cam')  # Camera
                self.r_x_dm = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_dm.shape, name='r_x_dm')
                self.r_x_cam = tf.placeholder(tf.float32, [None] + self.placeholders_configs.inputs.x_cam.shape, name='r_x_cam')
                self.is_training = False

            self.x_dm_ft = tf.identity(self.x_dm[:, :, :, 3:5], name='x_dm_ft')
            self.r_x_dm_ft = tf.identity(self.r_x_dm[:, :, :, 3:5], name='r_x_dm_ft')
            self.y_hat_se3param = stereo_rggnet_forward(
                x_dm_ft=self.x_dm_ft, r_x_dm_ft=self.r_x_dm_ft,
                x_cam=self.x_cam, r_x_cam=self.r_x_cam, is_training=self.is_training)

    def define_graph(self):
        self.logger.info('[InferenceModel] Constructing graph now...')
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.define_net()


class InferenceCkptProcessor(object):

    def process(self, config_fp, from_dir, to_dir, to_name):
        config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
        config = Box(yaml.safe_load(open(config_abs_fp, 'r').read()))
        # Config logger
        tf.logging.set_verbosity(tf.logging.INFO)
        logger = tf.logging

        inference_model = InferenceModel(config=config, logger=logger)
        logger.info("Create inference graph from {} to {}".format(from_dir, to_dir))
        create_inference_graph(inference_model.graph, from_dir, to_dir, to_name, config.inference.included_tensor_names)
        freeze_graph_from_file(to_dir, to_name, config.inference.freeze.output_node_name)


if __name__ == '__main__':
    fire.Fire(InferenceCkptProcessor)
