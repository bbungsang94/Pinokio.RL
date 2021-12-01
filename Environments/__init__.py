from functools import partial
from Environments.AGV_Dispatching.AGV_Dispatching import MultiAgentEnv, AGVBased
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {"sc2": partial(env_fn, env=AGVBased)}

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
