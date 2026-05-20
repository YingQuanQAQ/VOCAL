from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from bootstrap import ensure_shared_training_path

ensure_shared_training_path()

from loss import policy_gradient_loss  # noqa: E402
from util import ops  # noqa: E402
from util.decorator import within_cuda_device  # noqa: E402
from util.misc import check_done  # noqa: E402


class Worker(mp.Process):
    def __init__(
        self,
        master,
        net,
        env,
        worker_id,
        gpu_id=0,
        t_max=20,
        gamma=0.99,
        tau=1.0,
        ae_loss_k=1.0,
        commit_loss_k=0.25,
        lang_loss_k=1.0,
        anchor_loss_k=1.0,
        anchor_loss_min=0.05,
        vqvib_beta=0.05,
    ):
        super().__init__()

        self.worker_id = worker_id
        self.net = net
        self.env = env
        self.master = master
        self.t_max = t_max
        self.gamma = gamma
        self.tau = tau
        self.gpu_id = gpu_id
        self.reward_log = deque(maxlen=5)
        self.sparse_reward_log = deque(maxlen=5)
        self.shaped_reward_log = deque(maxlen=5)
        self.pfmt = (
            "policy loss: {} value loss: {} entropy loss: {} "
            "commit loss: {} lang loss: {} anchor loss: {} "
            "lambda_anchor: {} reward: {}"
        )
        self.agents = ["agent_{}".format(i) for i in range(self.env.num_agents)]
        self.num_acts = 1
        self.ae_loss_k = ae_loss_k
        self.commit_loss_k = commit_loss_k
        self.lang_loss_k = lang_loss_k
        self.anchor_loss_k = anchor_loss_k
        self.anchor_loss_min = anchor_loss_min
        self.vqvib_beta = vqvib_beta

    def get_anchor_weight(self, weight_iter):
        if self.master.max_iteration <= 0:
            return self.anchor_loss_min
        progress = min(float(weight_iter) / float(self.master.max_iteration), 1.0)
        return self.anchor_loss_k - (self.anchor_loss_k - self.anchor_loss_min) * progress

    def get_model_device(self):
        return next(self.net.parameters()).device

    def get_base_env(self):
        return getattr(self.env, "unwrapped", self.env)

    def _adjacent(self, p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1

    def _is_adjacent_to_any(self, pos, positions):
        for q in positions:
            if self._adjacent(pos, q):
                return True
        return False

    def _held_entity_idx(self, player):
        if player.held_object is None:
            return 3  # Nothing
        name = player.held_object.name
        if name == "onion":
            return 0
        if name == "dish":
            return 1
        if name == "soup":
            return 2
        return 3

    def _infer_location_status_idx(self, mdp, state, player, held_idx):
        pos = tuple(player.position)
        onion_disp = mdp.get_onion_dispenser_locations()
        dish_disp = mdp.get_dish_dispenser_locations()
        pots = mdp.get_pot_locations()
        serving = mdp.get_serving_locations()

        if held_idx == 0:  # onion
            if self._is_adjacent_to_any(pos, pots):
                return 1  # to the Pot
            if self._is_adjacent_to_any(pos, onion_disp):
                return 0  # from the Dispenser
            return 3  # at the Counter

        if held_idx == 1:  # plate
            if self._is_adjacent_to_any(pos, dish_disp):
                return 0
            if self._is_adjacent_to_any(pos, pots):
                return 1
            return 3

        if held_idx == 2:  # soup
            if self._is_adjacent_to_any(pos, serving):
                return 2  # to the Serving Station
            return 2

        # empty hands
        if self._is_adjacent_to_any(pos, onion_disp) or self._is_adjacent_to_any(pos, dish_disp):
            return 0  # from the Dispenser
        if self._is_adjacent_to_any(pos, pots):
            pot_states = mdp.get_pot_states(state)
            if pot_states.get("cooking"):
                return 1  # near a cooking pot -> "to the Pot"
        return 3

    def _infer_action_intent_idx(self, mdp, state, player, held_idx):
        pos = tuple(player.position)
        pots = mdp.get_pot_locations()
        if held_idx == 3 and self._is_adjacent_to_any(pos, pots):
            pot_states = mdp.get_pot_states(state)
            if pot_states.get("cooking"):
                return 2  # waiting for
        if held_idx == 3:
            return 0  # picking up
        return 1  # putting down

    def get_slot_targets(self):
        if self.net.method != "vocal":
            return None
        base_env = self.get_base_env()
        mdp = base_env._base_env.mdp
        state = base_env._base_env.state

        slot_targets = {}
        for agent_id, agent_name in enumerate(self.agents):
            player = state.players[agent_id]
            held_idx = self._held_entity_idx(player)
            intent_idx = self._infer_action_intent_idx(mdp, state, player, held_idx)
            loc_idx = self._infer_location_status_idx(mdp, state, player, held_idx)
            slot_targets[agent_name] = [intent_idx, held_idx, loc_idx]

        for agent_name, slot_target in slot_targets.items():
            slot_targets[agent_name] = self.net.comm_ae.format_slot_targets(
                slot_target)
        return slot_targets

    def get_lang_ground_loss(self, vocal_info, slot_target, agent_name):
        if self.net.method != "vocal" or slot_target is None:
            return torch.tensor(0.0, device=self.get_model_device())
        agent_idx = int(agent_name.split("_")[-1])
        slot_anchors = self.net.comm_ae.get_text_anchors()
        slot_losses = []
        for slot_idx, target_idx in enumerate(slot_target):
            pred = vocal_info["z_slots"][slot_idx][agent_idx].unsqueeze(0)
            target = slot_anchors[slot_idx][target_idx].unsqueeze(0)
            slot_losses.append(1.0 - F.cosine_similarity(pred, target, dim=-1).mean())
        return torch.stack(slot_losses).mean()

    @within_cuda_device
    def get_trajectory(self, hidden_state, state_var, done):
        env_mask_idx = [None for _ in range(len(self.agents))]
        trajectory = [[] for _ in range(self.num_acts)]

        while not check_done(done) and len(trajectory[0]) < self.t_max:
            slot_targets = self.get_slot_targets()
            plogit, value, hidden_state, comm_out, vocal_info = self.net(
                state_var, hidden_state, env_mask_idx=env_mask_idx
            )
            action, _, _, all_actions = self.net.take_action(plogit, comm_out)
            state, reward, done, info = self.env.step(all_actions)
            state_var = ops.to_state_var(state)

            trajectory[0].append(
                (plogit, action, value, reward, info, vocal_info, slot_targets)
            )

            for agent_id, a in enumerate(self.agents):
                if info[a]["done"] and env_mask_idx[agent_id] is None:
                    env_mask_idx[agent_id] = [0, 1, 2, 3]

        if check_done(done):
            target_value = [{k: 0 for k in self.agents} for _ in range(self.num_acts)]
        else:
            with torch.no_grad():
                target_value = self.net(state_var, hidden_state, env_mask_idx=env_mask_idx)[1]
                if self.num_acts == 1:
                    target_value = [target_value]

        values = [{k: None for k in self.agents} for _ in range(self.num_acts)]
        for k in self.agents:
            for aid in range(self.num_acts):
                values[aid][k] = [x[k] for x in list(zip(*trajectory[aid]))[2]]
                next_value = target_value[aid][k]
                if torch.is_tensor(next_value):
                    values[aid][k].append(next_value.detach())
                else:
                    values[aid][k].append(ops.to_torch([next_value]))
                values[aid][k].reverse()

        return trajectory, values, target_value, done

    @within_cuda_device
    def run(self):
        self.master.init_tensorboard()
        done = True
        reward_log = 0.0
        sparse_reward_log = 0.0
        shaped_reward_log = 0.0
        latest_episode_reward = 0.0
        latest_episode_sparse_reward = 0.0
        latest_episode_shaped_reward = 0.0

        while not self.master.is_done():
            weight_iter = self.master.copy_weights(self.net)
            self.net.zero_grad()

            if check_done(done):
                state = self.env.reset()
                state_var = ops.to_state_var(state)
                hidden_state = None
                if self.net.is_recurrent:
                    hidden_state = self.net.init_hidden()
                done = False

                self.reward_log.append(reward_log)
                self.sparse_reward_log.append(sparse_reward_log)
                self.shaped_reward_log.append(shaped_reward_log)
                latest_episode_reward = reward_log
                latest_episode_sparse_reward = sparse_reward_log
                latest_episode_shaped_reward = shaped_reward_log
                reward_log = 0.0
                sparse_reward_log = 0.0
                shaped_reward_log = 0.0

            trajectory, values, target_value, done = self.get_trajectory(
                hidden_state, state_var, done
            )

            all_pls = [[] for _ in range(self.num_acts)]
            all_vls = [[] for _ in range(self.num_acts)]
            all_els = [[] for _ in range(self.num_acts)]

            recon_losses = []
            commit_losses = []
            lang_losses = []
            lambda_anchor = 0.0 if self.net.method != "vocal" else self.get_anchor_weight(weight_iter)
            vqvib_total_losses = []
            vqvib_kl_losses = []
            vqvib_entropy_losses = []
            vqvib_recon_losses = []

            loss_a3c = 0
            for aid in range(self.num_acts):
                traj = trajectory[aid]
                val = values[aid]
                tar_val = target_value[aid]
                traj.reverse()

                for agent in self.agents:
                    gae = torch.zeros(1, 1, device=self.get_model_device())
                    t_value = tar_val[agent]

                    pls, vls, els = [], [], []
                    for i, (pi_logit, action, value, reward, info, vocal_info, slot_targets) in enumerate(traj):
                        if self.net.method == "ae-comm":
                            recon_losses.append(vocal_info["recon_loss"])
                        elif self.net.method == "vqvib":
                            vqvib_total_losses.append(vocal_info["vq_total_loss"])
                            vqvib_kl_losses.append(vocal_info["vq_kl_loss"])
                            vqvib_entropy_losses.append(vocal_info["vq_entropy_loss"])
                            vqvib_recon_losses.append(vocal_info["vq_recon_loss"])
                        else:
                            commit_loss = F.mse_loss(vocal_info["z"], vocal_info["c_k"].detach())
                            commit_losses.append(commit_loss)
                            lang_losses.append(
                                self.get_lang_ground_loss(vocal_info, slot_targets[agent], agent)
                            )

                        t_value = reward[agent] + self.gamma * t_value
                        advantage = t_value - value[agent]
                        delta_t = reward[agent] + self.gamma * val[agent][i].data - val[agent][i + 1].data
                        gae = gae * self.gamma * self.tau + delta_t

                        tl, (pl, vl, el) = policy_gradient_loss(
                            pi_logit[agent], action[agent], advantage, gae=gae
                        )
                        pls.append(ops.to_numpy(pl))
                        vls.append(ops.to_numpy(vl))
                        els.append(ops.to_numpy(el))
                        reward_log += reward[agent]
                        sparse_reward_log += info[agent]["sparse_reward"]
                        shaped_reward_log += info[agent]["shaped_reward"]
                        loss_a3c += tl

                    all_pls[aid].append(np.mean(pls))
                    all_vls[aid].append(np.mean(vls))
                    all_els[aid].append(np.mean(els))

            loss_recon = torch.stack(recon_losses).mean() if recon_losses else loss_a3c.new_tensor(0.0)
            loss_commit = torch.stack(commit_losses).mean() if commit_losses else loss_a3c.new_tensor(0.0)
            loss_lang = torch.stack(lang_losses).mean() if lang_losses else loss_a3c.new_tensor(0.0)
            loss_vqvib = torch.stack(vqvib_total_losses).mean() if vqvib_total_losses else loss_a3c.new_tensor(0.0)
            loss_vqvib_kl = torch.stack(vqvib_kl_losses).mean() if vqvib_kl_losses else loss_a3c.new_tensor(0.0)
            loss_vqvib_entropy = torch.stack(vqvib_entropy_losses).mean() if vqvib_entropy_losses else loss_a3c.new_tensor(0.0)
            loss_vqvib_recon = torch.stack(vqvib_recon_losses).mean() if vqvib_recon_losses else loss_a3c.new_tensor(0.0)

            if self.net.method == "vocal":
                loss_anchor = self.net.comm_ae.get_anchor_loss()
                anchor_distance = self.net.comm_ae.get_anchor_distance()
                loss = (
                    loss_a3c
                    + self.commit_loss_k * loss_commit
                    + self.lang_loss_k * loss_lang
                    + lambda_anchor * loss_anchor
                )
            elif self.net.method == "vqvib":
                loss_anchor = loss_a3c.new_tensor(0.0)
                anchor_distance = loss_a3c.new_tensor(0.0)
                loss_lang = loss_a3c.new_tensor(0.0)
                loss = loss_a3c + loss_vqvib
            else:
                loss_anchor = loss_a3c.new_tensor(0.0)
                anchor_distance = loss_a3c.new_tensor(0.0)
                loss_lang = loss_a3c.new_tensor(0.0)
                loss = loss_a3c + self.ae_loss_k * loss_recon

            loss.backward()

            if self.worker_id == 0:
                log_dict = {}
                for act_id, act in enumerate(["env", "comm"][: self.num_acts]):
                    for agent_id, agent in enumerate(self.agents):
                        log_dict["{}_policy_loss/{}".format(act, agent)] = all_pls[act_id][agent_id]
                        log_dict["{}_value_loss/{}".format(act, agent)] = all_vls[act_id][agent_id]
                        log_dict["{}_entropy/{}".format(act, agent)] = all_els[act_id][agent_id]
                    log_dict["policy_loss/{}".format(act)] = float(np.mean(all_pls[act_id]))
                    log_dict["value_loss/{}".format(act)] = float(np.mean(all_vls[act_id]))
                    log_dict["entropy/{}".format(act)] = float(np.mean(all_els[act_id]))

                if self.net.method == "ae-comm":
                    log_dict["ae_loss"] = ops.to_numpy(loss_recon)
                elif self.net.method == "vqvib":
                    log_dict["vqvib/loss_total"] = ops.to_numpy(loss_vqvib)
                    log_dict["vqvib/loss_kl"] = ops.to_numpy(loss_vqvib_kl)
                    log_dict["vqvib/loss_entropy"] = ops.to_numpy(loss_vqvib_entropy)
                    log_dict["vqvib/loss_recon"] = ops.to_numpy(loss_vqvib_recon)
                else:
                    log_dict["loss_commit"] = ops.to_numpy(loss_commit)
                    log_dict["loss_lang"] = ops.to_numpy(loss_lang)
                    log_dict["loss_anchor"] = ops.to_numpy(loss_anchor)
                    log_dict["lambda_anchor"] = lambda_anchor
                    if weight_iter % 100 == 0:
                        log_dict["Comm/Anchor_Distance"] = ops.to_numpy(anchor_distance)

                log_dict["train/reward_recent5"] = float(np.mean(self.reward_log))
                log_dict["train/reward_latest_episode"] = float(latest_episode_reward)
                log_dict["train/sparse_reward_recent5"] = float(np.mean(self.sparse_reward_log))
                log_dict["train/sparse_reward_latest_episode"] = float(latest_episode_sparse_reward)
                log_dict["train/shaped_reward_recent5"] = float(np.mean(self.shaped_reward_log))
                log_dict["train/shaped_reward_latest_episode"] = float(latest_episode_shaped_reward)

                for k, v in log_dict.items():
                    self.master.writer.add_scalar(k, v, weight_iter)

            progress_str = self.pfmt.format(
                np.around(np.mean(all_pls, axis=-1), decimals=5),
                np.around(np.mean(all_vls, axis=-1), decimals=5),
                np.around(np.mean(all_els, axis=-1), decimals=5),
                np.around(ops.to_numpy(
                    loss_recon if self.net.method == "ae-comm"
                    else loss_vqvib if self.net.method == "vqvib"
                    else loss_commit), decimals=5),
                np.around(ops.to_numpy(loss_lang), decimals=5),
                np.around(ops.to_numpy(loss_anchor), decimals=5),
                np.around(lambda_anchor, decimals=5),
                np.around(np.mean(self.reward_log), decimals=2),
            )

            self.master.apply_gradients(self.net)
            self.master.increment(progress_str)

        print("worker {} is done.".format(self.worker_id))
