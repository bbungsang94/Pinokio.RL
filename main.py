import platform
import random
import numpy as np
import torch as th
from utils.logging import get_logger
import utils.config_mapper as config_mapper

'''
algorithm 설정 가이드(config/algs 경로의 파일이름 그대로)
만일 rnn agent의 QMIX 를 실행하고 싶다면 -> 'RNN_AGENT/qmix_beta'
만일 G2ANet agent COMA 를 실행하고 싶다면 -> 'G2ANet_Agent/coma'
만일 ROMA 를 실행하고 싶다면 -> 'Role_Learning_Agent/qmix_smac_latent'
'''


if __name__ == '__main__':
    if platform.system() != 'Windows':
        print('This system is optimized on Winodws. Check your operating system(OS).')

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

    is_offline_run = config['off_pg']

    if is_offline_run:
        offpg_run(config, logger, env_name)
    else:
        standard_run(config, logger, env_name)