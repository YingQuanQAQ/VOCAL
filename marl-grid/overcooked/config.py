from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import datetime
import json
import os
from types import SimpleNamespace


def get_env_cfg():
    cfg = SimpleNamespace()
    cfg.seed = 1
    cfg.env_type = "overcooked"
    cfg.layout_name = "cramped_room"
    cfg.num_agents = 2
    cfg.max_steps = 400
    cfg.observation_style = "dict"
    cfg.observe_self_position = True
    cfg.observe_self_env_act = True
    cfg.comm_len = 10
    cfg.discrete_comm = True
    cfg.use_shaped_reward = True
    cfg.shaped_reward_scale = 1.0
    return cfg


def get_env_name(env_cfg):
    name = "Overcooked-{}".format(env_cfg.layout_name)
    if env_cfg.comm_len > 0:
        name += "-{}C".format(env_cfg.comm_len)
        if not env_cfg.discrete_comm:
            name += "cont"
    name += "-H{}".format(env_cfg.max_steps)
    return name


def get_default_cfg(args):
    cfg = SimpleNamespace()
    cfg.env_cfg = get_env_cfg()
    cfg.run_dir = "runs"
    cfg.num_workers = 2
    cfg.gpu = [int(g) for g in args.gpu]
    cfg.id = ""
    cfg.tmax = 20
    cfg.train_iter = 500000
    cfg.lr = 0.0001
    cfg.resume_path = ""
    cfg.method = args.method
    cfg.ablation = args.ablation
    cfg.ae_loss_k = 1.0
    cfg.commit_loss_k = 0.25
    cfg.lang_loss_k = 1.0
    cfg.anchor_loss_k = 1.0
    cfg.anchor_loss_min = 0.05
    cfg.slot_grouping = "task"
    cfg.ae_pg = 0
    cfg.ae_type = ""
    cfg.img_feat_dim = 64
    cfg.vqvib_num_protos = 16
    cfg.vqvib_beta = 0.05
    cfg.vqvib_commit_alpha = 0.25
    cfg.vqvib_entropy_weight = 0.02
    cfg.vqvib_recons_weight = 1.0
    cfg.use_evaluator = False
    cfg.use_multiprocessing = False
    return cfg


def get_config(args):
    cfg = get_default_cfg(args)
    apply_overrides(cfg, args.set)
    apply_ablation(cfg)
    cfg.env_cfg.env_name = get_env_name(cfg.env_cfg)

    curr_time = str(datetime.datetime.now())[:16].replace(" ", "_")
    id_args = [
        ["seed", cfg.env_cfg.seed],
        ["method", cfg.method],
        ["lr", cfg.lr],
        ["tmax", cfg.tmax],
        ["workers", cfg.num_workers],
        ["ms", cfg.env_cfg.max_steps],
        ["layout", cfg.env_cfg.layout_name],
    ]
    cfg_id = "_".join(["{}-{}".format(k, v) for k, v in id_args])
    if cfg.id:
        cfg_id = "{}_{}".format(cfg.id, cfg_id)
    if cfg.ablation != "full":
        cfg_id = "{}_abl-{}".format(cfg_id, cfg.ablation)
    cfg.exp_name = "{}/a3c_{}_{}".format(cfg.env_cfg.env_name, cfg_id, curr_time)
    return cfg


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+")
    parser.add_argument("--gpu", nargs="+", default=["0"])
    parser.add_argument("--method", type=str, default="vocal",
                        choices=["ae-comm", "vocal", "vqvib", "no-comm"])
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no-slot", "no-commit",
                                 "no-lang", "no-anchor"])
    args = parser.parse_args()
    cfg = get_config(args)
    freeze(cfg, save_file=True)
    print("Registered env [{}]".format(cfg.env_cfg.env_name))
    print("------ Overcooked A3C configurations ------")
    print("gpu: {}".format(args.gpu))
    print("method: {}".format(args.method))
    print("-------------------------------------------")
    return cfg


def freeze(config, save_file=False):
    if not save_file:
        return
    os.makedirs(config.run_dir, exist_ok=True)
    save_dir = os.path.join(config.run_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    payload = namespace_to_dict(config)
    with open(os.path.join(save_dir, "config.json"), "w") as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    return obj


def apply_overrides(cfg, overrides):
    if not overrides:
        return
    if len(overrides) % 2 != 0:
        raise ValueError("--set expects key value pairs")

    for key, value in zip(overrides[::2], overrides[1::2]):
        target = cfg
        attr_name = key
        if "." in key:
            prefix, attr_name = key.split(".", 1)
            if prefix != "env_cfg":
                raise ValueError("Only env_cfg.* or top-level keys are supported")
            target = cfg.env_cfg
        current = getattr(target, attr_name)
        if isinstance(current, bool):
            parsed = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int):
            parsed = int(value)
        elif isinstance(current, float):
            parsed = float(value)
        else:
            parsed = value
        setattr(target, attr_name, parsed)


def apply_ablation(cfg):
    if cfg.ablation == "full":
        return
    if cfg.method != "vocal":
        raise ValueError("Ablations are only supported for method=vocal")

    if cfg.ablation == "no-slot":
        cfg.slot_grouping = "single"
    elif cfg.ablation == "no-commit":
        cfg.commit_loss_k = 0.0
    elif cfg.ablation == "no-lang":
        cfg.lang_loss_k = 0.0
    elif cfg.ablation == "no-anchor":
        cfg.anchor_loss_k = 0.0
        cfg.anchor_loss_min = 0.0
    else:
        raise ValueError("Unknown ablation: {}".format(cfg.ablation))
