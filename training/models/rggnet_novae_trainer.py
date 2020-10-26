from ..core.base_trainer import BaseTrainer as BaseTrainerTemplate


class BaseTrainer(BaseTrainerTemplate):

    def pre_process(self, feed_dict, tensor_dict):
        # Normalization
        feed_dict[tensor_dict['x_cam']] = feed_dict[tensor_dict['x_cam']] / 255.
        return feed_dict
