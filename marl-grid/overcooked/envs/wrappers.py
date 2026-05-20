try:
    import gym
except ImportError:  # pragma: no cover
    import gymnasium as gym


class DictObservationNormalizationWrapper(gym.Wrapper):
    """Normalize uint8 image observations to the range [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs_dict, rew_dict, done_dict, info_dict = self.env.step(action)
        self._normalize_obs(obs_dict)
        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        self._normalize_obs(obs_dict)
        return obs_dict

    def _normalize_obs(self, obs_dict):
        for key, value in obs_dict.items():
            if key == "global" or not isinstance(value, dict):
                continue
            if "pov" in value:
                value["pov"] = 2.0 * ((value["pov"] / 255.0) - 0.5)

    def __getattr__(self, name):
        return getattr(self.env, name)
