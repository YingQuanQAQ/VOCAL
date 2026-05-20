from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import sys
from glob import glob

import numpy as np

try:
    import gym
except ImportError:  # pragma: no cover
    import gymnasium as gym

from bootstrap import path_matches_current_python


def _ensure_overcooked_src():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(
        os.path.join(this_dir, "..", "..", "..", "..")
    )
    venv_candidates = glob(
        os.path.join(
            workspace_root,
            "overcooked_ai-master",
            ".venv",
            "lib",
            "python*",
            "site-packages",
        )
    )
    for site_packages in venv_candidates:
        if not path_matches_current_python(site_packages):
            continue
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
    src_dir = os.path.join(workspace_root, "overcooked_ai-master", "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_ensure_overcooked_src()

from overcooked_ai_py.mdp.actions import Action  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402


class OvercookedMultiAgentEnv(gym.Env):
    """Wrap Overcooked-AI with the marl-grid multi-agent dict interface."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, env_cfg):
        super().__init__()

        self.layout_name = env_cfg.layout_name
        self.horizon = env_cfg.max_steps
        self.num_agents = env_cfg.num_agents
        self.comm_len = env_cfg.comm_len
        self.discrete_comm = env_cfg.discrete_comm
        self.observe_self_position = env_cfg.observe_self_position
        self.observe_self_env_act = env_cfg.observe_self_env_act
        self.obs_dtype = np.uint8
        self.env_name = env_cfg.env_name
        self.use_shaped_reward = getattr(env_cfg, "use_shaped_reward", True)
        self.shaped_reward_scale = getattr(env_cfg, "shaped_reward_scale", 1.0)

        if self.num_agents != 2:
            raise ValueError("Overcooked currently supports exactly 2 agents")

        self._base_env = self._build_base_env()
        self.visualizer = None
        self.adv_indices = []
        self.step_count = 0

        self.last_env_actions = np.full(
            self.num_agents, Action.ACTION_TO_INDEX[Action.STAY], dtype=np.int64
        )
        self.last_comm_actions = self._zero_comm_actions()

        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _build_base_env(self):
        mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        return OvercookedEnv.from_mdp(mdp, horizon=self.horizon, info_level=0)

    def _zero_comm_actions(self):
        if self.discrete_comm:
            return np.zeros((self.num_agents, self.comm_len), dtype=np.int64)
        return np.zeros((self.num_agents, self.comm_len), dtype=np.float32)

    def _build_observation_space(self):
        pov_shape = tuple(self._base_env.mdp.get_lossless_state_encoding_shape())
        width, height = self._base_env.mdp.shape

        obs_space = {
            "pov": gym.spaces.Box(
                low=0,
                high=255,
                shape=pov_shape,
                dtype=self.obs_dtype,
            )
        }

        if self.observe_self_position:
            obs_space["selfpos"] = gym.spaces.MultiDiscrete([width, height])

        if self.observe_self_env_act:
            obs_space["self_env_act"] = gym.spaces.Discrete(n=Action.NUM_ACTIONS)

        if self.comm_len > 0:
            if self.discrete_comm:
                per_agent_comm = [
                    gym.spaces.MultiDiscrete([2 for _ in range(self.comm_len)])
                    for _ in range(self.num_agents)
                ]
                obs_space["comm"] = gym.spaces.Tuple(per_agent_comm)
            else:
                obs_space["comm"] = gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_agents, self.comm_len),
                    dtype=np.float32,
                )

        return gym.spaces.Dict(obs_space)

    def _build_action_space(self):
        env_action_space = gym.spaces.Discrete(n=Action.NUM_ACTIONS)
        if self.comm_len <= 0:
            return env_action_space

        if self.discrete_comm:
            comm_action_space = gym.spaces.MultiDiscrete(
                [2 for _ in range(self.comm_len)]
            )
        else:
            comm_action_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.comm_len,), dtype=np.float32
            )
        return gym.spaces.Tuple((env_action_space, comm_action_space))

    def _lossless_uint8_obs(self):
        encodings = self._base_env.lossless_state_encoding_mdp(self._base_env.state)
        obs = []
        for encoding in encodings:
            scaled = np.clip(encoding, 0.0, 1.0) * 255.0
            obs.append(scaled.astype(self.obs_dtype))
        return obs

    def _get_comm_obs(self, agent_idx):
        if self.comm_len <= 0:
            return None
        others = [idx for idx in range(self.num_agents) if idx != agent_idx]
        ordered_indices = others + [agent_idx]
        return np.stack([self.last_comm_actions[idx] for idx in ordered_indices], axis=0)

    def _build_agent_obs(self):
        obs_dict = {}
        pov_obs = self._lossless_uint8_obs()

        for agent_idx in range(self.num_agents):
            player = self._base_env.state.players[agent_idx]
            agent_obs = {"pov": pov_obs[agent_idx]}

            if self.observe_self_position:
                agent_obs["selfpos"] = np.asarray(
                    player.position, dtype=np.int64
                )

            if self.observe_self_env_act:
                agent_obs["self_env_act"] = np.int64(
                    self.last_env_actions[agent_idx]
                )

            comm_obs = self._get_comm_obs(agent_idx)
            if comm_obs is not None:
                agent_obs["comm"] = comm_obs

            obs_dict["agent_{}".format(agent_idx)] = agent_obs

        obs_dict["global"] = {
            "t": np.float32(self.step_count / max(self.horizon, 1)),
            "env_act": self.last_env_actions.copy(),
            "comm_act": self.last_comm_actions.copy(),
        }
        return obs_dict

    def _extract_actions(self, action_dict):
        env_actions = []
        comm_actions = self._zero_comm_actions()

        for agent_idx in range(self.num_agents):
            action = action_dict["agent_{}".format(agent_idx)]
            if self.comm_len > 0:
                env_act = int(action[0])
                comm_act = np.asarray(action[1])
                if self.discrete_comm:
                    comm_act = comm_act.astype(np.int64)
                else:
                    comm_act = np.clip(comm_act.astype(np.float32), 0.0, 1.0)
                comm_actions[agent_idx] = comm_act
            else:
                env_act = int(action)
            env_actions.append(Action.INDEX_TO_ACTION[env_act])

        self.last_env_actions = np.asarray(
            [Action.ACTION_TO_INDEX[a] for a in env_actions], dtype=np.int64
        )
        self.last_comm_actions = comm_actions
        return tuple(env_actions)

    def _build_reward_dict(self, sparse_reward, shaped_rewards):
        reward_dict = {}
        for agent_idx in range(self.num_agents):
            train_reward = float(sparse_reward)
            if self.use_shaped_reward:
                train_reward += self.shaped_reward_scale * float(shaped_rewards[agent_idx])
            reward_dict["agent_{}".format(agent_idx)] = train_reward
        return reward_dict

    def _build_info(self, reward, done, env_info, reward_dict):
        sparse_reward = float(reward)
        shaped_rewards = env_info.get("shaped_r_by_agent", [0.0] * self.num_agents)

        info_dict = {}
        for agent_idx in range(self.num_agents):
            player = self._base_env.state.players[agent_idx]
            held_object = (
                player.held_object.name if player.held_object is not None else ""
            )
            info_dict["agent_{}".format(agent_idx)] = {
                "done": bool(done),
                "nonadv_done": bool(done),
                "comm": self.last_comm_actions[agent_idx].copy(),
                "comm_str": "",
                "posd": np.asarray(
                    [player.position[0], player.position[1], int(done)],
                    dtype=np.int64,
                ),
                "held_object": held_object,
                "sparse_reward": sparse_reward,
                "shaped_reward": float(shaped_rewards[agent_idx]),
                "train_reward": float(reward_dict["agent_{}".format(agent_idx)]),
            }

        info_dict["rew_by_act"] = {
            0: {
                "agent_{}".format(i): sparse_reward for i in range(self.num_agents)
            },
            1: {
                "agent_{}".format(i): sparse_reward for i in range(self.num_agents)
            },
        }
        info_dict["episode_sparse_reward"] = sparse_reward
        info_dict["episode_shaped_reward_by_agent"] = [
            float(r) for r in shaped_rewards
        ]
        info_dict["reward_mode"] = {
            "use_shaped_reward": bool(self.use_shaped_reward),
            "shaped_reward_scale": float(self.shaped_reward_scale),
        }
        info_dict["layout_name"] = self.layout_name
        return info_dict

    def reset(self):
        self._base_env.reset()
        self.step_count = 0
        self.last_env_actions = np.full(
            self.num_agents, Action.ACTION_TO_INDEX[Action.STAY], dtype=np.int64
        )
        self.last_comm_actions = self._zero_comm_actions()
        return self._build_agent_obs()

    def step(self, action_dict):
        joint_action = self._extract_actions(action_dict)
        _, reward, done, env_info = self._base_env.step(joint_action)
        self.step_count = self._base_env.state.timestep

        obs_dict = self._build_agent_obs()
        shaped_rewards = env_info.get("shaped_r_by_agent", [0.0] * self.num_agents)
        reward_dict = self._build_reward_dict(reward, shaped_rewards)
        done_dict = {"__all__": bool(done)}
        info_dict = self._build_info(reward, done, env_info, reward_dict)
        return obs_dict, reward_dict, done_dict, info_dict

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError("Only rgb_array rendering is supported")
        return self.get_raw_obs()

    def get_raw_obs(self):
        if self.visualizer is None:
            import pygame
            from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
            self._pygame = pygame
            self.visualizer = StateVisualizer()

        rewards_dict = {
            "cumulative_sparse_rewards_by_agent": (
                self._base_env.game_stats["cumulative_sparse_rewards_by_agent"]
            ),
            "cumulative_shaped_rewards_by_agent": (
                self._base_env.game_stats["cumulative_shaped_rewards_by_agent"]
            ),
        }
        image = self.visualizer.render_state(
            state=self._base_env.state,
            grid=self._base_env.mdp.terrain_mtx,
            hud_data=self.visualizer.default_hud_data(
                self._base_env.state, **rewards_dict
            ),
        )
        frame = self._pygame.surfarray.array3d(image)
        return np.flip(np.rot90(frame, 3), 1)

    def __deepcopy__(self, memo):
        del memo
        new_env = self.__class__(copy.deepcopy(self._build_env_cfg()))
        new_env._base_env = self._base_env.copy()
        new_env._base_env.reset(regen_mdp=False)
        new_env._base_env.state = self._base_env.state.deepcopy()
        new_env.step_count = self.step_count
        new_env.last_env_actions = self.last_env_actions.copy()
        new_env.last_comm_actions = self.last_comm_actions.copy()
        return new_env

    def _build_env_cfg(self):
        class _Cfg(object):
            pass

        cfg = _Cfg()
        cfg.layout_name = self.layout_name
        cfg.max_steps = self.horizon
        cfg.num_agents = self.num_agents
        cfg.comm_len = self.comm_len
        cfg.discrete_comm = self.discrete_comm
        cfg.observe_self_position = self.observe_self_position
        cfg.observe_self_env_act = self.observe_self_env_act
        cfg.env_name = self.env_name
        cfg.use_shaped_reward = self.use_shaped_reward
        cfg.shaped_reward_scale = self.shaped_reward_scale
        return cfg
