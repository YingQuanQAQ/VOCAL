from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp

import config
from bootstrap import ensure_shared_training_path, maybe_reexec_into_overcooked_venv

maybe_reexec_into_overcooked_venv()
ensure_shared_training_path()

import torch
import torch.multiprocessing as mp

from envs.environments import make_environment
from no_comm_network import NoCommNetwork

from actor_critic.master import Master  # noqa: E402
from actor_critic.worker import Worker  # noqa: E402
from util.shared_opt import SharedAdam  # noqa: E402


if __name__ == "__main__":
    start_method = "fork" if os.name != "nt" else "spawn"
    print(torch.multiprocessing.get_start_method())
    mp.set_start_method(start_method, force=True)
    os.environ["OMP_NUM_THREADS"] = "1"

    cfg = config.parse()
    cfg.env_cfg.comm_len = 0
    cfg.env_cfg.discrete_comm = False
    cfg.env_cfg.env_name = config.get_env_name(cfg.env_cfg)
    cfg.method = "no-comm"

    save_dir_fmt = osp.join("./{}".format(cfg.run_dir), cfg.exp_name + "/{}")
    print(">> {}".format(cfg.exp_name))

    create_env = lambda: make_environment(cfg.env_cfg)
    env = create_env()
    create_net = lambda: NoCommNetwork(
        obs_space=env.observation_space,
        action_size=env.action_space.n,
        num_agents=cfg.env_cfg.num_agents,
        img_feat_dim=cfg.img_feat_dim,
    )

    master_lock = mp.Lock()
    net = create_net()
    net.share_memory()
    opt = SharedAdam(net.parameters(), lr=cfg.lr)

    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path, map_location="cpu")
        global_iter = mp.Value("i", ckpt["iter"])
        net.load_state_dict(ckpt["net"])
        opt.load_state_dict(ckpt["opt"])
        print(">>>>> Loaded ckpt from iter", ckpt["iter"])
    else:
        global_iter = mp.Value("i", 0)
    global_done = mp.Value("i", 0)

    master = Master(
        net, opt, global_iter, global_done, master_lock,
        writer_dir=save_dir_fmt.format("tb"),
        max_iteration=cfg.train_iter
    )

    workers = []
    for worker_id in range(cfg.num_workers):
        gpu_id = cfg.gpu[worker_id % len(cfg.gpu)]
        print("(worker {}) initializing on gpu {}".format(worker_id, gpu_id))
        workers.append(
            Worker(
                master,
                create_net(),
                create_env(),
                worker_id=worker_id,
                gpu_id=gpu_id,
                t_max=cfg.tmax,
                num_acts=1,
            )
        )

    if cfg.use_multiprocessing:
        [w.start() for w in workers]
        [w.join() for w in workers]
    else:
        assert cfg.num_workers == 1, (
            "single-process mode currently supports exactly one worker"
        )
        workers[0].run()

    ckpt_dir = save_dir_fmt.format("ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    master.save_ckpt(cfg.train_iter, osp.join(ckpt_dir, "latest.pth"))
