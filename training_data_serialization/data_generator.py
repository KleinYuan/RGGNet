import os
import fire
import yaml
import pathlib
from box import Box
from kitti_tools.data_raw import load_raw_data
from kitti_tools.data_training import generate_single_example
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from random import shuffle


class Generator(object):

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def run(self, config_fp):
        # Load configurations
        print("Loading configurations ...")
        _config = Box(yaml.load(open(config_fp, 'r').read()))
        print("Configs: \n{}".format(_config))

        for _dataset_name in _config.keys():
            _dataset_config = _config[_dataset_name]
            # Parse Configs
            _raw_data_dir = _dataset_config.raw_data_dir
            _output_data_dir = _dataset_config.output_data_dir
            _num_data = _dataset_config.number
            _range_rot = _dataset_config.off_range.rot
            _range_trans = _dataset_config.off_range.trans
            _vae_resized_H = _dataset_config.vae_resized.H
            _vae_resized_W = _dataset_config.vae_resized.W
            if 'force_shape' in _dataset_config:
                force_shape = (_dataset_config.force_shape.H, _dataset_config.force_shape.W)
            else:
                force_shape = None

            if 'reduce_lidar_line_to' in _dataset_config:
                reduce_lidar_line_to = _dataset_config.reduce_lidar_line_to
            else:
                reduce_lidar_line_to = None

            # Load all data
            print("Loading data ...")
            _all_raw_data = load_raw_data(_raw_data_dir)

            # Generate data
            _all_training_raw_data = self.get_training_raw_data(all_raw_data=_all_raw_data, num_data=_num_data)

            # Serialize as tfrecords
            self.serialize_into_tfrecords(training_raw_data=_all_training_raw_data, output_fp=_output_data_dir,
                                          file_size=int(_num_data / 10), off_range_rot=_range_rot, off_range_trans=_range_trans,
                                          vae_resized_H=_vae_resized_H, vae_resized_W=_vae_resized_W, force_shape=force_shape,
                                          reduce_lidar_line_to=reduce_lidar_line_to)

    def get_training_raw_data(self, all_raw_data, num_data):
        """
        Directing pre-compute all training data will make your memory explode
        So we only pre-compute the raw data s for training which is by default not loaded
        """
        print("Generating training raw data objects............")
        num_raw_data = len(all_raw_data)
        _iters = num_data // num_raw_data + 1

        training_raw_data = []
        for _ in range(0, _iters):
            for _idx, _one_raw_data in enumerate(all_raw_data):
                training_raw_data.append(_one_raw_data)

        # shuffle
        shuffle(training_raw_data)
        return training_raw_data[:num_data]

    def serialize_into_tfrecords(self, training_raw_data, output_fp, file_size, off_range_rot, off_range_trans, vae_resized_H, vae_resized_W, force_shape=None, reduce_lidar_line_to=None):
        _training_raw_data_chunks = self.chunks(lst=training_raw_data, n=file_size)

        print("Creating {} tfrecord with each including {} examples............".format(len(training_raw_data) / file_size, file_size))
        for _chunk_idx, _training_raw_data_chunk in tqdm(enumerate(_training_raw_data_chunks)):
            if not os.path.isdir(output_fp):
                print("{} does not exits, creating one.".format(output_fp))
                pathlib.Path(output_fp).mkdir(parents=True, exist_ok=True)
            writer = tf.python_io.TFRecordWriter("{}/{}.tfrecord".format(output_fp, _chunk_idx))
            for _idx, _one_raw_data in enumerate(_training_raw_data_chunk):
                _training_data = generate_single_example(_one_raw_data, off_range_rot, off_range_trans, vae_resized_H,
                                                         vae_resized_W, force_shape, reduce_lidar_line_to=reduce_lidar_line_to)

                example_dict = {}
                example_dict.update({
                    'x_dm': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.x_dm.flatten().astype(np.float32))),
                    'x_cam': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.x_cam.flatten().astype(np.float32))),
                    'x_R_rects': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.x_R_rects.flatten().astype(np.float32))),
                    'x_P_rects': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.x_P_rects.flatten().astype(np.float32))),
                    'y_dm': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.y_dm.flatten().astype(np.float32))),
                    'y_se3param': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.y_se3param.flatten().astype(np.float32))),
                    'x_cam_resized': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.x_cam_resized.flatten().astype(np.float32))),
                    'x_dm_ft_resized': tf.train.Feature(float_list=tf.train.FloatList(value=_training_data.x_dm_ft_resized.flatten().astype(np.float32))),
                })
                example = tf.train.Example(features=tf.train.Features(feature=example_dict))
                writer.write(example.SerializeToString())

            writer.close()
            print("Created {}.{}.tfrecord".format(output_fp, _chunk_idx))


if __name__ == '__main__':
    fire.Fire(Generator)
