import os

import yaml
from fvcore.common.config import CfgNode


def dir_creation(path):
    if not os.path.exists(path):
        os.makedirs(path)



def get_cfg(config_path) -> CfgNode:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = CfgNode(cfg)

    return cfg


class ModelFactory:
    @staticmethod
    def create_model(task_name, class_name, **kwargs):
        # import class dynamically
        module = __import__('scripts.' + task_name + '.' + class_name, fromlist=[class_name])
        model_class = getattr(module, class_name)

        return model_class(**kwargs)

