import numpy as np
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def stub_feed_dict(tensor_dict, replace_none_as=1, mode='train'):
    print("Generate stub feed_dict ....")
    feed_dict = {}
    for _tensor_name, _content in tensor_dict.items():
        # TODO: We are using the protocol here that a non-batch data is wrapped up with a dict
        if type(_content) == dict:
            feed_dict[_content['tensor']] = _content['value'][mode]
            print("   {} is assigned with a stub data with {}".format(_content['tensor'], _content['value'][mode]))
        else:
            _shape_list = [replace_none_as if v is None else v for v in _content.get_shape().as_list()]
            feed_dict[_content] = np.random.rand(*_shape_list).astype(_content.dtype.as_numpy_dtype)
            print("   {} is assigned with a stub data with shape of {} and dtype of {}".format(_tensor_name, feed_dict[_content].shape, feed_dict[_content].dtype))
    return feed_dict
