import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from ..models.rggnet_model import Model as RggNetModel
from ..models.vae_model import Model as VAEModel
from box import Box
import yaml
from ..tests import libs

model_map = {
    'rggnet': {
        'model': RggNetModel,
        'gpu_only': False
    }
    ,
    'vae': {
        'model': VAEModel,
        'gpu_only': False
    }
}


def _run_case(model_name):
    if model_map[model_name]['gpu_only'] and len(libs.get_available_gpus()) == 0:
        print("Test requires GPU however not available! Skip Test.")
        return
    config_fp = '../config/{}.yaml'.format(model_name)
    config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
    config = Box(yaml.load(open(config_abs_fp, 'r').read()))
    logger = tf.logging
    model = model_map[model_name]['model'](config=config, logger=logger)
    print("Model: {}".format(model))
    tf_config = tf.ConfigProto(allow_soft_placement=True, device_count=config.train.devices)
    tf_config.gpu_options.allow_growth = True
    with model.graph.as_default():
        with tf.Session(config=tf_config) as sess:
            feed_dict = libs.stub_feed_dict(tensor_dict=model.tensor_dict, replace_none_as=config.train.batch_size)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            loss_v = sess.run(model.loss, feed_dict=feed_dict)
    assert loss_v is not None, "Test Failed!"


def test_models():
    for _model_name, _model_ins in model_map.items():
        print("Testing {} ...".format(_model_name))
        tf.reset_default_graph()
        _run_case(model_name=_model_name)

