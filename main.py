import platform
import random
import numpy as np
import torch as th
from utils.logging import get_logger
import utils.config_mapper as config_mapper
from Runners.offpg_run import offpg_run
from Runners.run import standard_run

'''
algorithm 설정 가이드(config/algs 경로의 파일이름 그대로)
'''


if __name__ == '__main__':
    if platform.system() != 'Windows':
        print('This system is optimized on Windows. Check your operating system(OS).')

    logger = get_logger()

    algorithm = 'RNN_AGENT/graphmix'
    env_name = 'AGV_Dispatching'
    map_name = 'acs_dda_simulator_210324'

    config = config_mapper.config_copy(
        config_mapper.get_config(algorithm=algorithm, env_name=env_name, map_name=map_name))

    random_Seed = random.randrange(0, 7777)
    np.random.seed(random_Seed)
    th.manual_seed(random_Seed)
    config['env_args']['seed'] = random_Seed
    config['env_args']['map_name'] = map_name

    is_offline_run = config['off_pg']

    if is_offline_run:
        offpg_run(config, logger, env_name)
    else:
        standard_run(config, logger, env_name)