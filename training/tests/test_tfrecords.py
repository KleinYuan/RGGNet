from ..utils.tfrecords import get_iterator_from_tfrecords
import os
import yaml
from box import Box
import tensorflow as tf
import numpy as np
import cv2

def stats(x):
    _h, _w, _c = x.shape
    for _i in range(0, _c):
        _x = x[:, :, _i]
        print("      Channel : {}".format(_i))
        print("            shape  : {}".format(_x.shape))
        print("            dtype  : {}".format(_x.dtype))
        print("            max    : {}".format(np.max(_x)))
        print("            min    : {}".format(np.min(_x)))
    print("      All Channel: ")
    print("            shape  : {}".format(x.shape))
    print("            dtype  : {}".format(x.dtype))
    print("            max    : {}".format(np.max(x)))
    print("            min    : {}".format(np.min(x)))


def test_rggnet_batch(config_fp='../config/rggnet.yaml',):
    config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
    config = Box(yaml.load(open(config_abs_fp, 'r').read()))
    print("Only fetching the first data ...")
    config.data.tfrecords_train_dirs = config.data.tfrecords_toy_dirs
    print("============================")
    print(config.data)
    print("============================")

    iterator, init_op = get_iterator_from_tfrecords(config, test=False)
    next = iterator.get_next()
    print("Next Keys: {}".format(next.keys()))
    x_dm = next['x_dm']
    x_cam = next['x_cam']
    x_R_rects = next['x_R_rects']
    x_P_rects = next['x_P_rects']
    # x_dm_ft_resized = next['x_dm_ft_resized']
    # x_cam_resized = next['x_cam_resized']

    y_dm = next['y_dm']
    y_se3param = next['y_se3param']
    cnt = 0
    with tf.Session() as sess:
        sess.run(init_op)
        try:
            while True:
                cnt += 1
                _x_dm, _x_cam, _x_R_rects, _x_P_rects,  _y_dm, _y_se3param = \
                    sess.run([x_dm, x_cam, x_R_rects, x_P_rects, y_dm, y_se3param])
                print("{}   y_se3param: {}".format(cnt, _y_se3param.shape))
        except Exception as e:
            print("End of data.")


def test_rggnet(config_fp='../config/rggnet.yaml',):
    config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
    config = Box(yaml.load(open(config_abs_fp, 'r').read()))
    print("Only fetching the first data ...")
    config.data.tfrecords_train_dirs = config.data.tfrecords_toy_dirs
    config.train.batch_size = 1
    print(config.data)

    iterator, init_op = get_iterator_from_tfrecords(config, test=False)
    next = iterator.get_next()
    print("Next Keys: {}".format(next.keys()))
    x_dm = next['x_dm']
    x_cam = next['x_cam']
    x_R_rects = next['x_R_rects']
    x_P_rects = next['x_P_rects']
    # x_dm_ft_resized = next['x_dm_ft_resized']
    # x_cam_resized = next['x_cam_resized']

    y_dm = next['y_dm']
    y_se3param = next['y_se3param']

    with tf.Session() as sess:
        sess.run(init_op)
        [_x_dm], [_x_cam], [_x_R_rects], [_x_P_rects],  [_y_dm], [_y_se3param] = \
            sess.run([x_dm, x_cam, x_R_rects, x_P_rects, y_dm, y_se3param])
        print("Images: ")
        print("   x_dm     : ")
        stats(_x_dm)
        print("   x_cam    : ")
        stats(_x_cam)
        print("   y_dm     : ")
        stats(_y_dm)

        x_dm_vis = np.concatenate([_x_dm[:, :, 3:5], _x_dm[:, :, 3:4]], -1) * 255
        y_dm_vis = np.concatenate([_y_dm[:, :, 3:5], _y_dm[:, :, 3:4]], -1) * 255
        x_dm_vis = cv2.addWeighted(x_dm_vis, 1.0, _x_cam, 0.7, 0)
        y_dm_vis = cv2.addWeighted(y_dm_vis, 1.0, _x_cam, 0.7, 0)
        img_vis = np.concatenate([x_dm_vis, y_dm_vis], 0)
        print("   y_se3param: {}".format(_y_se3param))

    cv2.imshow('img_vis', img_vis.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_vae(config_fp='../config/vae.yaml',):
    config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
    config = Box(yaml.load(open(config_abs_fp, 'r').read()))
    print("Only fetching the first data ...")
    config.data.tfrecords_train_dirs = config.data.tfrecords_toy_dirs
    config.train.batch_size = 1
    print(config.data)

    iterator, init_op = get_iterator_from_tfrecords(config, test=False)
    next = iterator.get_next()
    print("Next Keys: {}".format(next.keys()))
    x_dm_ft_resized = next['x_dm_ft_resized']
    x_cam_resized = next['x_cam_resized']

    with tf.Session() as sess:
        sess.run(init_op)
        try:
            while True:
                [_x_dm_ft_resized], [_x_cam_resized] = sess.run([x_dm_ft_resized, x_cam_resized])
                print("Images: ")
                print("   x_dm_ft_resized     : ")
                stats(_x_dm_ft_resized)
                print("   _x_cam_resized    : ")
                stats(_x_cam_resized)

                x_dm_ft_resized_vis = np.concatenate([_x_dm_ft_resized, _x_dm_ft_resized[:, :, 0:1]], -1) * 255
                img_vis = cv2.addWeighted(x_dm_ft_resized_vis, 1.0, _x_cam_resized, 0.3, 0)

                cv2.imshow('img_vis', img_vis.astype(np.uint8))
                cv2.waitKey(0)
        except Exception:
            print("End of data!")
            cv2.destroyAllWindows()
            exit()


if __name__ == "__main__":
    test_rggnet_batch()
    test_rggnet()
    test_vae()
