import collections
from copy import deepcopy
import yaml


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_config(algorithm, env_name, map_name):
    config_dir = '{0}/{1}'
    config_dir2 = '{0}/{1}/{2}'

    with open(config_dir.format('config', "{}.yaml".format('default')), "r") as f:
        try:
            default_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    with open(config_dir2.format('config', 'environments', "{}.yaml".format(env_name)), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format('sc2', exc)
        env_config = config_dict

    with open(config_dir2.format('config', 'algorithms', "{}.yaml".format(algorithm)), "r") as f:
        try:
            config_dict2 = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format('sc2', exc)
        alg_config = config_dict2

    final_config_dict = recursive_dict_update(default_config, env_config)
    final_config_dict = recursive_dict_update(final_config_dict, alg_config)

    final_config_dict['dbms_arg']['map_name'] = map_name

    return final_config_dict