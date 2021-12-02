
class DataBaseMap:
    agvs = 3
    ttt = 4


map_param_registry = {
    "acs_dda_simulator_210324": {
        "host": '127.0.0.1',
        "user": 'root',
        "password": '151212kyhASH@',
        "port": 3306,
        "limit": 3000,
    },
    "default": {
        "host": '127.0.0.1',
        "user": 'root',
        "password": '151212kyhASH@',
        "port": 3306,
        "limit": 3000,
    },
}


def get_database_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (DataBaseMap,), dict(filename=name))
