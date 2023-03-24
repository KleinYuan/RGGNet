# This app is to efficiently to test against tfrecords with frozen protobuf

import os
import fire
import yaml
import pathlib
from box import Box
from src.pb_server import PBServer
from env import Env


class Evaluator(object):

    def run_all(self, config_fp, profile_export=None):
        # This functions pass the same initial input to all three graphs
        model_names = ['rggnet', 'rggnet_novae']
        given_data = []
        for _id, _model_name in enumerate(model_names):
            if _id == 0:
                all_inference_data, all_training_obj = self.run(config_fp, _model_name, profile_export=profile_export, keep_data=True)
                given_data = [all_inference_data, all_training_obj]
            else:
                self.run(config_fp, _model_name, profile_export=profile_export, given_data=given_data)

    def run(self, config_fp, model_name, profile_export=None, keep_data=False, given_data=None):
        print("Running evaluation of {} with given data ? {}".format(model_name, given_data is not None))
        # Loading Config
        config_abs_fp = os.path.join(os.path.dirname(__file__), config_fp)
        config = Box(yaml.safe_load(open(config_abs_fp, 'r').read()))
        # Load environment
        env = Env(config=config.env, model_name=model_name)
        app_config = config.app[model_name]
        # Loading Inference Server
        inference_server = PBServer(config=app_config)

        if profile_export is not None:
            profile_export = profile_export + '/' + model_name
            if not os.path.isdir(profile_export):
                print("{} does not exits, creating one.".format(profile_export))
                pathlib.Path(profile_export).mkdir(parents=True, exist_ok=True)

        all_inference_data = []
        all_training_obj = []

        if env.fs_mode():
            all_raw_data = env.prepare_all_testing_data()
            if given_data is None:
                for _id, _example_raw_data in enumerate(all_raw_data):
                    _inference_data, _training_obj = env.prepare_one_inference_data(_example_raw_data)
                    if keep_data:
                        all_inference_data.append(_inference_data)
                        all_training_obj.append(_training_obj)
                    assert len(_inference_data) == 2
                    assert len(_inference_data[0]) == len(_inference_data[1])
                    if profile_export is not None:
                        _profile_export = profile_export + '/{}'.format(_id)
                    else:
                        _profile_export = None
                    _y_hat_se3param = inference_server.inference(data=_inference_data, profile_export=_profile_export)
                    env.write_report(example=_training_obj, pred=_y_hat_se3param)
            else:
                _given_inference_data, _given_training_obj = given_data
                for _id, _inference_data in enumerate(_given_inference_data):
                    assert len(_inference_data) == 2
                    assert len(_inference_data[0]) == len(_inference_data[1])
                    if profile_export is not None:
                        _profile_export = profile_export + '/{}'.format(_id)
                    else:
                        _profile_export = None
                    _y_hat_se3param = inference_server.inference(data=_inference_data, profile_export=_profile_export)
                    env.write_report(example=_given_training_obj[_id], pred=_y_hat_se3param)

        if keep_data:
            return all_inference_data, all_training_obj


if __name__ == '__main__':
    fire.Fire(Evaluator)
