from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import config
from envs.environments import make_environment


def sample_action(env):
    return {
        "agent_{}".format(agent_idx): env.action_space.sample()
        for agent_idx in range(env.num_agents)
    }


def main():
    cfg = config.get_config(type("Args", (), {"set": None})())
    env = make_environment(cfg.env_cfg)

    obs = env.reset()
    print("Reset ok. Keys:", sorted(obs.keys()))
    for agent_idx in range(env.num_agents):
        agent_key = "agent_{}".format(agent_idx)
        pov = obs[agent_key]["pov"]
        print(
            "{} pov_shape={} selfpos={} self_env_act={}".format(
                agent_key,
                pov.shape,
                obs[agent_key].get("selfpos"),
                obs[agent_key].get("self_env_act"),
            )
        )

    for step_idx in range(5):
        action = sample_action(env)
        obs, reward, done, info = env.step(action)
        print(
            "step={} reward={} done={} held={}".format(
                step_idx,
                reward,
                done["__all__"],
                [info["agent_{}".format(i)]["held_object"] for i in range(env.num_agents)],
            )
        )
        if done["__all__"]:
            break

    frame = env.get_raw_obs()
    print("Render ok. frame_shape={}".format(np.asarray(frame).shape))


if __name__ == "__main__":
    main()
