from Environments import REGISTRY as env_REGISTRY
from Environments.Tetris import Tetris_basic
import utils.config_mapper as config_mapper

algorithm = 'RNN_AGENT/qmix'
env_name = 'Tetris_multi'
map_name = 'acs_dda_simulator_210324'

config = config_mapper.config_copy(
    config_mapper.get_config(algorithm=algorithm, env_name=env_name, map_name=map_name))

Env = env_REGISTRY[env_name](**config['env_args'])
Env.reset()

Env.step(0)
