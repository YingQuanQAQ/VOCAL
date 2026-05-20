# Overcooked Wrapper

This directory contains an isolated Overcooked-AI environment wrapper for the
existing `marl-grid` training interface.

## Goals

- Do not modify `find-goal` or `red-blue-doors`
- Reuse the current multi-agent dict observation interface
- Keep Overcooked dependencies isolated in
  `/home/root123/HQ/LanEmerg/overcooked_ai-master/.venv`

## Implemented

- `envs/overcooked_environment.py`
  - wraps Overcooked joint actions into the existing `action_dict` format
  - exposes per-agent observations with `pov`, `selfpos`, `self_env_act`, `comm`
  - exposes `reward`, `done`, `info`, `get_raw_obs`, and `render`
- `envs/environments.py`
  - provides `make_environment(env_cfg)`
- `config.py`
  - lightweight environment and training configuration
- `smoke_test_env.py`
  - validates `reset`, `step`, render, and deepcopy compatibility
- `ae_network.py`
  - Overcooked-specific AE / VQ communication network
- `worker_ae.py`
  - Overcooked-specific worker with heuristic three-slot `vocal` supervision
- `train_ae.py`
  - training entrypoint for `ae-comm` and `vocal`

## Observation Design

- `pov`
  - uses Overcooked lossless state encoding for each player
  - is converted to `uint8` and normalized to `[-1, 1]` by the wrapper
- `selfpos`
  - agent's current `(x, y)` position
- `self_env_act`
  - previous environment action index
- `comm`
  - ordered as `other agents first, self last`, matching the existing marl-grid convention

## Smoke Test

Run with the isolated Overcooked environment:

```bash
cd /home/root123/HQ/LanEmerg/overcooked_ai-master
.venv/bin/python /home/root123/HQ/LanEmerg/marl-ae-comm/marl-grid/overcooked/smoke_test_env.py
```

## Training

Run from the current `marl-ae-comm` training environment:

```bash
cd /home/root123/HQ/LanEmerg/marl-ae-comm/marl-grid/overcooked
python train_ae.py --method ae-comm --gpu 0 --set num_workers 1 train_iter 500000 env_cfg.max_steps 400
python train_ae.py --method vocal --gpu 0 --set num_workers 1 train_iter 500000 env_cfg.max_steps 400
```

Notes:

- the default Overcooked training path reuses `torch` from the current
  `marl-ae-comm` environment
- Overcooked-specific dependencies such as `gymnasium` and `pygame` are loaded
  from `/home/root123/HQ/LanEmerg/overcooked_ai-master/.venv`
- the current training entrypoint defaults to single-process mode for stability
  with Overcooked's environment object on Linux

## Status

This wrapper now supports:

- environment reset / step / render / deepcopy
- `ae-comm` training entry
- `vocal` training entry with heuristic three-slot grounding targets

The next improvement would be replacing the current heuristic slot targets with
stronger task-phase labels derived from richer pot / counter / teammate intent
logic.
