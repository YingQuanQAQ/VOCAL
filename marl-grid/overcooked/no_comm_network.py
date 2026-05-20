from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from bootstrap import ensure_shared_training_path

ensure_shared_training_path()

from model.a3c_template import A3CTemplate  # noqa: E402
from model.init import normalized_columns_initializer  # noqa: E402
from model.model_utils import LSTMhead  # noqa: E402

from ae_network import InputProcessor


class NoCommNetwork(A3CTemplate):
    def __init__(self, obs_space, action_size, num_agents, hidden_size=256,
                 img_feat_dim=64):
        super().__init__()
        self.action_size = action_size
        self.num_agents = num_agents
        self.input_processor = InputProcessor(
            obs_space, 0, num_agents, last_fc_dim=img_feat_dim
        )
        self.feat_dim = self.input_processor.feat_dim

        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size, num_layers=1)
             for _ in range(num_agents)]
        )
        self.is_recurrent = True

        self.critic_linear = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_agents)]
        )
        self.actor_linear = nn.ModuleList(
            [nn.Linear(hidden_size, self.action_size) for _ in range(num_agents)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.actor_linear:
            m.weight.data = normalized_columns_initializer(m.weight.data, 0.01)
            m.bias.data.fill_(0)
        for m in self.critic_linear:
            m.weight.data = normalized_columns_initializer(m.weight.data, 1.0)
            m.bias.data.fill_(0)

    def init_hidden(self):
        return [head.init_hidden() for head in self.head]

    def take_action(self, policy_logit):
        act_dict = {}
        act_logp_dict = {}
        ent_list = []
        for agent_name, logits in policy_logit.items():
            act, act_logp, ent = super(NoCommNetwork, self).take_action(logits)
            act_dict[agent_name] = act
            act_logp_dict[agent_name] = act_logp
            ent_list.append(ent)
        return act_dict, act_logp_dict, ent_list

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1

        cat_feat = self.input_processor(inputs)
        actor_out, critic_out = {}, {}
        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == "global":
                continue
            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i])
            actor_out[agent_name] = self.actor_linear[i](x)
            critic_out[agent_name] = self.critic_linear[i](x)
            if env_mask_idx and env_mask_idx[i]:
                actor_out[agent_name][0, env_mask_idx[i]] = -1e10
        return actor_out, critic_out, hidden_state
