
class DataBaseMap:
    agvs = 3
    ttt = 4


map_param_registry = {
    "bane_vs_bane": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 200,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "bane",
    },
    "2c_vs_64zg": {
        "n_agents": 2,
        "n_enemies": 64,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },
}


def get_database_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (DataBaseMap,), dict(filename=name))
