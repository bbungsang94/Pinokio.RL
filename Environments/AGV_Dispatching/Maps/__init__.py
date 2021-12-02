from . import LGmaps


def get_map_params(map_name):
    map_param_registry = LGmaps.get_database_map_registry()
    return map_param_registry[map_name]
