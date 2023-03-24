"""
Author: Kaiwen

Customized Architecture of Neural Network -----
                                              |
                                              |
                                              |
Customized Loss/Optimizer -----------------------> BaseCompute -----> [["ops": ops, "pls": pls]..]

"""

import tensorflow as tf


class BaseModel(object):
    """
    Basic model that defines the generic structure of model.
    You are expected to override follow implementations:

    - init (where you define a bunch of properties before define_graph)
    - define_loss (where you shall ensure you define self.loss)
    - define_optimizer (where you shall ensure you define self.optimizer)
    - define_net (where you construct tensorflow ops into graph)
    - define_tensor_dict (where you define self.tensor_dict)
    """

    def init(self, *args, **kwargs):
        """
        Adding your customized init here
        :return:
        """
        raise NotImplementedError

    def define_loss(self, *args, **kwargs):
        """
        Put your loss here
        """
        raise NotImplementedError

    def define_optimizer(self, *args, **kwargs):
        """
        Put your optimizer here
        """
        raise NotImplementedError

    def define_net(self, *args, **kwargs):
        """
        Put your neural network architecture here
        """
        raise NotImplementedError

    def define_tensor_dict(self, *args, **kwargs):
        """
        Define a dictionary such as following:
        self.tensor_dict = {
            'X': self.x_pl,
            'Y': self.y_pl,
            # Below are training flags tensors such as is_training, keep_prob
            # That you need to ensure it contains tensor/value (train/inference) keys
            'is_training':
                {
                    'tensor': self.is_training,
                    'value': {
                        'train': True,
                        'inference': False
                    }
                },
            'keep_prob':
                {
                    'tensor': self.keep_prob,
                    'value': {
                        'train': self.config.tensors.hyper_params.keep_prob,
                        'inference': 1.0
                    }
                }
        }
        """
        raise NotImplementedError

    def __init__(self, config, logger, *args, **kwargs):

        # Below we define the interface
        self.tensor_dict = None
        self.init_graph = None
        self.optimizer = None
        self.graph = None
        self.loss = None

        # Below are optional
        self.restore_graph_fns = []
        self.summary_list = []
        self.logger = logger
        self.config = config
        self.start_lr = float(config.train.start_learning_rate)
        self.decay_steps = float(config.train.lr_decay_step)
        self.decay_rate = float(config.train.lr_decay_rate)
        self.init()

        self.define_graph()

    def check_model(self, model, variables_to_restore):
        print('@#@#$@#$@#$@#$ !!!!!!! model: ', model)
        # print('var: ', variables_to_restore[0].name)
        variables_in_resnet = tf.train.list_variables(model)
        variable_map = {}
        # print(variables_in_resnet)
        # print(variables_to_restore)
        for var in variables_in_resnet:
            from_var_name = var[0]

            for dest_name in ['left_cam', 'right_cam']:#, 'left_dm', 'right_dm']:
                to_var_name = from_var_name.replace('resnet_v2_50', dest_name)
                for my_var in variables_to_restore:
                    if my_var.name.split(':')[0] == to_var_name:
                        variable_map[from_var_name] = my_var

            # to_var_name = from_var_name.replace('resnet_v2_50', 'right_cam')
            # for my_var in variables_to_restore:
            #     if my_var.name.split(':')[0] == to_var_name:
            #         variable_map[from_var_name] = my_var

        # print('var: ', variable_map)
        return variable_map

    def get_restore_weights_fn(self, model_dir=None, ckpts_dir=None, included_tensor_names=None, excluded_tensor_names=None):
        """
        https://www.tensorflow.org/api_docs/python/tf/contrib/framework/get_variables_to_restore
        :param dir: Directory which you save the pretrained checkpoints: .meta, .data, .index
        :param included_tensor_names: the tensor names you wanna include to restore
        :param excluded_tensor_names: the tensor names you wanna exclude to restore
        :return: a function that's supposed to load all params into memory
        """
        self.logger.info("[Restoring weights] Setting ......")
        if ckpts_dir is None and model_dir is None:
            raise Exception("Neither model_dir nor ckpts_dir is provided!")
        elif ckpts_dir is not None:
            checkpoint = tf.train.get_checkpoint_state(ckpts_dir)
            model_dir = checkpoint.model_checkpoint_path
        self.logger.info("[Restoring weights] From {}".format(model_dir))
        with self.graph.as_default():
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(
                include=included_tensor_names,
                exclude=excluded_tensor_names)
            # print('Variables to restore:', variables_to_restore)
            variable_map = self.check_model(model_dir, variables_to_restore)
            restore_graph_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_dir, variable_map,
                                                                              ignore_missing_vars=True)
            # restore_graph_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_dir, variables_to_restore,
                                                                            #   ignore_missing_vars=True)
            print("!@#!@#!@#!@# Restore graph fn: ", restore_graph_fn, included_tensor_names)

        self.logger.info("[Restoring weights] {} variables will be restored from {}".format(len(variables_to_restore), model_dir))
        return restore_graph_fn

    def define_restore_graph_fns(self):
        print("[BaseCompute] Define restore graph fns ...")
        fn = []
        if len(self.config.train.pre_trained_weights) > 0:
            print('Pretrained-Weights: ', self.config.train.pre_trained_weights)
            for _pre_trained_weights in self.config.train.pre_trained_weights:
                if _pre_trained_weights.load:
                    _excluded_tensor_names = None \
                        if len(_pre_trained_weights.excluded_tensor_names) == 0 \
                        else _pre_trained_weights.excluded_tensor_names
                    _included_tensor_names = None \
                        if len(_pre_trained_weights.included_tensor_names) == 0 \
                        else _pre_trained_weights.included_tensor_names
                    if 'ckpts_dir' in _pre_trained_weights:
                        f = self.get_restore_weights_fn(
                            ckpts_dir=_pre_trained_weights.ckpts_dir,
                            excluded_tensor_names=_excluded_tensor_names,
                            included_tensor_names=_included_tensor_names)
                        if f:
                            fn.append(f)
                    elif 'model_dir' in _pre_trained_weights:
                        f = self.get_restore_weights_fn(
                            model_dir=_pre_trained_weights.model_dir,
                            excluded_tensor_names=_excluded_tensor_names,
                            included_tensor_names=_included_tensor_names)
                        if f:
                            fn.append(f)
                    else:
                        raise Exception("Neither ckpts_dir nor model_dir found in the config")
        if len(fn) <= 0:
            raise Exception("Failed to restore weights to graph.")
        self.restore_graph_fns = fn
        print('Restored Graph Fns: ', self.restore_graph_fns)

    def define_graph(self):
        self.logger.info('[BaseCompute] Constructing graph now...')
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.define_net()
            self.define_tensor_dict()
            # This must be executed before loss/optimizer defined
            self.define_loss()
            # Saving GPU memory
            with tf.device("/cpu:0"):
                self.define_restore_graph_fns()
            self.define_optimizer()
            self.define_summary_list()
            self.init_graph = tf.global_variables_initializer()

            # Check interface
            assert self.tensor_dict is not None, "self.tensor_dict is None!"
            assert self.init_graph is not None, "self.init_graph is None!"
            assert self.optimizer is not None, "self.optimizer is None!"
            assert self.graph is not None, "self.graph is None!"
            assert self.loss is not None, "self.loss is None!"

        self.logger.info('[BaseCompute] Graph constructed!')

    def define_summary_list(self):
        self.summary_list = [
            tf.summary.scalar("loss", self.loss)
        ]

    def get_summary_list(self):
        return self.summary_list
