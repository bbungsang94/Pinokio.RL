from . import acs_dda_simulator_210324


def get_map_params(map_name):
    map_param_registry = acs_dda_simulator_210324.get_database_map_registry()
    return map_param_registry[map_name]
