import importlib
import os
MODEL_NAME = "rggnet_novae"
training_app = importlib.import_module("training.apps.{}_training".format(MODEL_NAME))
config_file = MODEL_NAME + '.yaml'
if __name__ == "__main__":
    training_app.run(os.path.join('../config', config_file))
