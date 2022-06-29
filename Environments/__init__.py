from functools import partial
from Environments.AGV_Dispatching.AGV_Dispatching import MultiAgentEnv, AGVBasedFeature
from Environments.Tetris.Tetris_basic import TetrisSingle, TetrisMulti
import sys
import os


# Using Pinokio.RL as PR

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {'AGV_Dispatching': partial(env_fn, env=AGVBasedFeature),
            'Tetris_single': partial(env_fn, env=TetrisSingle),
            'Tetris_multi': partial(env_fn, env=TetrisMulti)}

if sys.platform == "linux":
    os.environ.setdefault("AGV_Path",
                          os.path.join(os.getcwd(), "3rdparty", "./"))
