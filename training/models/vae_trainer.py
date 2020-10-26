from ..core.base_trainer import BaseTrainer as BaseTrainerTemplate
import numpy as np


class BaseTrainer(BaseTrainerTemplate):

    def pre_process(self, feed_dict, tensor_dict):
        feed_dict[tensor_dict['x_cam_resized']] = feed_dict[tensor_dict['x_cam_resized']] / 255.
        assert not np.any(np.isnan(feed_dict[tensor_dict['x_cam_resized']]))
        assert not np.any(np.isnan(feed_dict[tensor_dict['x_dm_ft_resized']]))
        return feed_dict
