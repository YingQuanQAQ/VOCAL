from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .overcooked_environment import OvercookedMultiAgentEnv
from .wrappers import DictObservationNormalizationWrapper


def make_environment(env_cfg, lock=None):
    """Create an isolated Overcooked environment wrapper."""
    del lock

    env = OvercookedMultiAgentEnv(env_cfg)
    env = DictObservationNormalizationWrapper(env)
    return env
