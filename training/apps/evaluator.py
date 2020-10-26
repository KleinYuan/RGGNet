# This app is to efficiently to test against tfrecords with frozen protobuf

import os
import fire
import yaml
from box import Box
import tensorflow as tf
import numpy as np
from ..core.base_pb_server import PBServer
from ..utils.tfrecords import get_iterator_from_tfrecords
os.environ['GEOMSTATS_BACKEND'] = 'numpy'  # NOQA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
from geomstats.special_euclidean_group import SpecialEuclideanGroup
import geomstats.lie_group as lie_group
import pathlib


class Evaluator(object):

    def cal_metrics(self, se3_pred, se3_true, group, metric):
        se3_error = lie_group.loss(se3_pred, se3_true, group, metric)
        return se3_error

    def process(self, config_fp, model_name, res_fp):
        # Loading Config
        config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
        config = Box(yaml.load(open(config_abs_fp, 'r').read()))
        config = config[model_name]
        print("Evaluating {}.".format(model_name))
        print("Config is  {}".format(config))

        # Loading Inference Server
        inference_server = PBServer(config=config)
        with inference_server.graph.as_default():
            # Loading TFRecords
            batch_iterator, batch_init_op = get_iterator_from_tfrecords(
                config=config,
                test=True)
            batch_next_op = batch_iterator.get_next()
        assert len(config.data.tfrecords_test_dirs) == 1, "Testing data larger than two :{}".format(config.data.tfrecords_test_dirs)
        test_name = config.data.tfrecords_test_dirs[0].split('/')[-1]
        print("Testing data name: {}".format(test_name))
        # Config metrics

        SE3_GROUP = SpecialEuclideanGroup(3, epsilon=np.finfo(np.float32).eps)
        metric = SE3_GROUP.left_canonical_metric

        # Loading Data Configurations
        cnt = 0
        se3_errors = []
        se3_noises = []
        se3_preds = []
        se3_gts = []
        RRs = []
        with inference_server.session as sess:
            sess.run(batch_init_op)
            try:
                while True:
                    _x_dm_batch, _x_cam_batch, _gt_se3param = sess.run([batch_next_op['x_dm'], batch_next_op['x_cam'], batch_next_op['y_se3param']])
                    _x_cam_batch = _x_cam_batch / 255.
                    _y_hat_se3param = inference_server.inference(data=[_x_dm_batch, _x_cam_batch])

                    se3_error = self.cal_metrics(
                        se3_pred=np.squeeze(np.array(_y_hat_se3param), 0),
                        se3_true=np.array(_gt_se3param),
                        group=SE3_GROUP,
                        metric=metric
                    )
                    se3_noise = self.cal_metrics(
                        se3_pred=np.zeros_like(np.array(_gt_se3param)),
                        se3_true=np.array(_gt_se3param),
                        group=SE3_GROUP,
                        metric=metric
                    )
                    RR = np.array(se3_error) / np.array(se3_noise)
                    RRs.append(RR)
                    MRR = 1 - np.mean(np.array(RRs))
                    if cnt == 1:
                        print("MRR Explain:")
                        print("RR = np.array(se3_error) / np.array(se3_noise)")
                        print("np.array(se3_error) = {}".format(np.array(se3_error)))
                        print("np.array(se3_noise) = {}".format(np.array(se3_noise)))
                        print("RR                   = {}".format(RR))
                        print("{} / {} = {}".format(np.array(se3_error)[0], np.array(se3_noise)[0], RR[0]))
                    se3_gts.append(np.array(_gt_se3param))
                    se3_preds.append(np.array(_y_hat_se3param))
                    se3_noises.append(se3_noise)
                    se3_noises_mean = np.mean(np.array(se3_noises))
                    se3_errors.append(se3_error)
                    se3_errors_mean = np.mean(np.array(se3_errors))
                    print("{} ~ {} Test --> {} | {} | {}%".format(
                        cnt*config.train.batch_size,
                        (1+cnt)*config.train.batch_size,
                        se3_errors_mean,
                        se3_noises_mean,
                        MRR * 100.))
                    cnt += 1

            except tf.errors.OutOfRangeError as e:
                print("End of data .")
        print("Final:  {} ({}%)".format(
            np.round(se3_errors_mean, 4),
            np.round(MRR * 100., 4)))

        if not os.path.isdir(res_fp):
            print("{} does not exits, creating one.".format(res_fp))
            pathlib.Path(res_fp).mkdir(parents=True, exist_ok=True)
        np.save('{}/{}_{}_se3_errors.npy'.format(res_fp, test_name, model_name), np.array(se3_errors))
        np.save('{}/{}_{}_se3_noises.npy'.format(res_fp, test_name, model_name), np.array(se3_noises))
        np.save('{}/{}_{}_se3_gt.npy'.format(res_fp, test_name, model_name), np.array(se3_gts))
        np.save('{}/{}_{}_se3_preds.npy'.format(res_fp, test_name, model_name), np.array(se3_preds))


if __name__ == '__main__':
    fire.Fire(Evaluator)
