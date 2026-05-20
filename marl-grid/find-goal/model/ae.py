from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.a3c_template import A3CTemplate, take_action, take_comm_action
from model.init import normalized_columns_initializer, weights_init
from model.model_utils import LSTMhead, ImgModule


class STE(torch.autograd.Function):
    """Straight-Through Estimator"""
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # clamp gradient between -1 and 1
        return F.hardtanh(grad_output)

FIND_GOAL_SLOT_NAMES = ['intent_entity', 'direction', 'distance']
FIND_GOAL_SLOT_VOCABS = [
    [
        'I found the goal',
        'I reached the goal and stopped',
        'I hit a wall / obstacle',
        'I am exploring blindly',
    ],
    [
        'to my North',
        'to my South',
        'to my East',
        'to my West',
        'to my North-East',
        'to my North-West',
        'to my South-East',
        'to my South-West',
        'right here',
    ],
    [
        'exactly 1 step away',
        'exactly 2 steps away',
        'exactly 3 steps away',
        'out of my sight',
    ],
]


class InputProcessor(nn.Module):
    """
    Pre-process the following individual observations:
        - pov (ImgModule)
        - self_env_act
        - selfpos
    """
    def __init__(self, obs_space, comm_feat_len, num_agents, last_fc_dim=64):
        super(InputProcessor, self).__init__()

        self.obs_keys = list(obs_space.spaces.keys())
        self.num_agents = num_agents

        # image processor
        assert 'pov' in self.obs_keys
        self.conv = ImgModule(obs_space['pov'].shape, last_fc_dim=last_fc_dim)
        feat_dim = last_fc_dim

        # state inputs processor
        state_feat_dim = 0

        if 'self_env_act' in self.obs_keys:
            # discrete value with one-hot encoding
            self.env_act_dim = obs_space.spaces['self_env_act'].n
            state_feat_dim += self.env_act_dim

        if 'selfpos' in self.obs_keys:
            self.discrete_positions = None
            if obs_space.spaces['selfpos'].__class__.__name__ == \
                    'MultiDiscrete':
                # process position with one-hot encoder
                self.discrete_positions = obs_space.spaces['selfpos'].nvec
                state_feat_dim += sum(self.discrete_positions)
            else:
                state_feat_dim += 2

        if state_feat_dim == 0:
            self.state_feat_fc = None
        else:
            # use state_feat_fc to process concatenated state inputs
            self.state_feat_fc = nn.Linear(state_feat_dim, 64)
            feat_dim += 64

        if self.state_feat_fc:
            self.state_layer_norm = nn.LayerNorm(64)
        self.img_layer_norm = nn.LayerNorm(last_fc_dim)

        # all other agents' decoded features, if provided
        self.comm_feat_dim = comm_feat_len * (num_agents - 1)
        feat_dim += self.comm_feat_dim

        self.feat_dim = feat_dim

    def forward(self, inputs, comm=None):
        # WARNING: the following code only works for Python 3.6 and beyond

        # process images together if provided
        if 'pov' in self.obs_keys:
            pov = []
            for i in range(self.num_agents):
                pov.append(inputs[f'agent_{i}']['pov'])
            x = torch.cat(pov, dim=0)
            x = self.conv(x)  # (N, img_feat_dim)
            xs = torch.chunk(x, self.num_agents)

        # concatenate observation features
        cat_feat = [self.img_layer_norm(xs[i]) for i in range(self.num_agents)]

        if self.state_feat_fc is None:
            if comm is not None:
                for i in range(self.num_agents):
                    # concat comm features for each agent
                    c = torch.reshape(comm[i], (1, self.comm_feat_dim))
                    cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)
            return cat_feat

        for i in range(self.num_agents):
            # concatenate state features
            feats = []

            # concat last env act if provided
            if 'self_env_act' in self.obs_keys:
                env_act = F.one_hot(
                    inputs[f'agent_{i}']['self_env_act'].to(torch.int64),
                    num_classes=self.env_act_dim)
                env_act = torch.reshape(env_act, (1, self.env_act_dim))
                feats.append(env_act)

            # concat agent's own position if provided
            if 'selfpos' in self.obs_keys:
                sp = inputs[f'agent_{i}']['selfpos'].to(torch.int64)  # (2,)
                if self.discrete_positions is not None:
                    spx = F.one_hot(sp[0],
                                    num_classes=self.discrete_positions[0])
                    spy = F.one_hot(sp[1],
                                    num_classes=self.discrete_positions[1])
                    sp = torch.cat([spx, spy], dim=-1).float()
                    sp = torch.reshape(sp, (1, sum(self.discrete_positions)))
                else:
                    sp = torch.reshape(sp, (1, 2))
                feats.append(sp)

            if len(feats) > 1:
                feats = torch.cat(feats, dim=-1)
            elif len(feats) == 1:
                feats = feats[0]
            else:
                raise ValueError('?!?!?!', feats)

            feats = self.state_feat_fc(feats)
            feats = self.state_layer_norm(feats)
            cat_feat[i] = torch.cat([cat_feat[i], feats], dim=-1)

            if comm is not None:
                # concat comm features for each agent
                c = torch.reshape(comm[i], (1, self.comm_feat_dim))
                cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)

        return cat_feat


class EncoderDecoder(nn.Module):
    def __init__(self, obs_space, comm_len, discrete_comm, num_agents,
                 ae_type='', img_feat_dim=64, method='vocal'):
        super(EncoderDecoder, self).__init__()

        self.preprocessor = InputProcessor(obs_space, 0, num_agents,
                                           last_fc_dim=img_feat_dim)
        in_size = self.preprocessor.feat_dim

        self.method = method
        self.discrete_comm = discrete_comm
        self.ae_type = ae_type
        self.comm_len = comm_len

        if self.method == 'ae-comm':
            if ae_type == 'rfc':
                self.encoder = nn.Sequential(
                    nn.Linear(in_size, comm_len),
                    nn.Sigmoid(),
                )
            elif ae_type == 'rmlp':
                self.encoder = nn.Sequential(
                    nn.Linear(in_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, comm_len),
                    nn.Sigmoid()
                )
            elif ae_type == 'fc':
                self.encoder = nn.Sequential(
                    nn.Linear(in_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, img_feat_dim),
                )
                self.fc = nn.Sequential(
                    nn.Linear(img_feat_dim, comm_len),
                    nn.Sigmoid(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(img_feat_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, in_size),
                )
            elif ae_type == 'mlp':
                self.encoder = nn.Sequential(
                    nn.Linear(in_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, img_feat_dim),
                )
                self.fc = nn.Sequential(
                    nn.Linear(img_feat_dim, img_feat_dim),
                    nn.ReLU(),
                    nn.Linear(img_feat_dim, img_feat_dim),
                    nn.ReLU(),
                    nn.Linear(img_feat_dim, comm_len),
                    nn.Sigmoid(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(img_feat_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, in_size),
                )
            elif ae_type == '':
                self.encoder = nn.Sequential(
                    nn.Linear(in_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, comm_len),
                    nn.Sigmoid()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(comm_len, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, in_size),
                )
            else:
                raise NotImplementedError
        elif self.method == 'vocal':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, comm_len)
            )
            self.slot_names = FIND_GOAL_SLOT_NAMES
            self.slot_vocabs = FIND_GOAL_SLOT_VOCABS
            self.slot_dims, slot_embeddings = self.load_slot_dictionary(
                'text_embeddings_find_goal.pt')
            self.num_slots = len(self.slot_dims)

            self.codebooks = nn.ParameterList()
            for slot_idx, embeddings in enumerate(slot_embeddings):
                self.register_buffer(f'text_anchor_{slot_idx}', embeddings)
                self.codebooks.append(nn.Parameter(embeddings.clone()))
        else:
            raise ValueError('Unknown method: {}'.format(self.method))

    def get_slot_dims(self, msg_dim, num_slots):
        base = msg_dim // num_slots
        remainder = msg_dim % num_slots
        slot_dims = [base for _ in range(num_slots)]
        for idx in range(remainder):
            slot_dims[idx] += 1
        return slot_dims

    def build_random_slot_embeddings(self, slot_vocabs, slot_dims):
        slot_embeddings = []
        for slot_vocab, slot_dim in zip(slot_vocabs, slot_dims):
            emb = torch.randn(len(slot_vocab), slot_dim)
            emb = F.normalize(emb, p=2, dim=1)
            slot_embeddings.append(emb)
        return slot_embeddings

    def load_slot_dictionary(self, file_name):
        anchor_path = os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))))),
            file_name)
        fallback_slot_dims = self.get_slot_dims(
            self.comm_len, len(self.slot_vocabs))

        try:
            payload = torch.load(anchor_path, map_location='cpu')
            slot_dims = payload['slot_dims']
            slot_embeddings = payload['slot_embeddings']
            slot_names = payload.get('slot_names')
            slot_vocabs = payload.get('slot_vocabs')

            if sum(slot_dims) != self.comm_len:
                raise ValueError('slot dims do not match comm_len')
            if len(slot_embeddings) != len(slot_dims):
                raise ValueError('slot embedding count does not match slot dims')
            if slot_names is not None:
                if len(slot_names) != len(slot_dims):
                    raise ValueError('slot names do not match slot dims')
                self.slot_names = slot_names
            if slot_vocabs is not None:
                if len(slot_vocabs) != len(slot_dims):
                    raise ValueError('slot vocabs do not match slot dims')
                for embeddings, vocab in zip(slot_embeddings, slot_vocabs):
                    if embeddings.shape[0] != len(vocab):
                        raise ValueError('slot vocab size does not match anchors')
                self.slot_vocabs = slot_vocabs
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            print('Warning: {} invalid or not found at {}, '
                  'initializing random slot anchors.'.format(
                      file_name, anchor_path))
            slot_dims = fallback_slot_dims
            slot_embeddings = self.build_random_slot_embeddings(
                self.slot_vocabs, slot_dims)

        return slot_dims, slot_embeddings

    def get_text_anchors(self):
        return [getattr(self, f'text_anchor_{idx}')
                for idx in range(len(self.codebooks))]

    def get_anchor_loss(self):
        losses = [
            F.mse_loss(codebook, anchor)
            for codebook, anchor in zip(self.codebooks, self.get_text_anchors())
        ]
        return torch.stack(losses).mean()

    def get_anchor_distance(self):
        distances = [
            torch.norm(codebook - anchor, dim=1).mean()
            for codebook, anchor in zip(self.codebooks, self.get_text_anchors())
        ]
        return torch.stack(distances).mean()

    def decode(self, x):
        """
        input: inputs[f'agent_{i}']['comm'] (num_agents, comm_len)
            (note that agent's own state is at the last index)
        """
        if self.method == 'vocal':
            return x

        if self.ae_type:
            return x
        return self.decoder(x)

    def forward(self, feat):
        if self.method == 'ae-comm':
            encoded = self.encoder(feat)

            if self.ae_type in {'rfc', 'rmlp'}:
                if self.discrete_comm:
                    encoded = STE.apply(encoded)
                return encoded, {'recon_loss': feat.new_tensor(0.0)}

            if self.ae_type in {'fc', 'mlp'}:
                decoded = self.decoder(encoded)
                loss = F.mse_loss(decoded, feat)
                comm = self.fc(encoded.detach())
                if self.discrete_comm:
                    comm = STE.apply(comm)
                return comm, {'recon_loss': loss}

            if self.ae_type == '':
                if self.discrete_comm:
                    encoded = STE.apply(encoded)
                decoded = self.decoder(encoded)
                loss = F.mse_loss(decoded, feat)
                return encoded.detach(), {'recon_loss': loss}

            raise NotImplementedError

        z = self.encoder(feat)
        z_slots = torch.split(z, self.slot_dims, dim=-1)

        quantized_slots = []
        codebook_slots = []
        slot_indices = []
        for z_slot, codebook in zip(z_slots, self.codebooks):
            z_expanded = z_slot.unsqueeze(1)
            codebook_expanded = codebook.unsqueeze(0)
            dist = torch.norm(z_expanded - codebook_expanded, dim=2)

            k_idx = torch.argmin(dist, dim=1)
            c_k = codebook[k_idx]
            c_k_ste = z_slot + (c_k - z_slot).detach()

            quantized_slots.append(c_k_ste)
            codebook_slots.append(c_k)
            slot_indices.append(k_idx)

        comm_out = torch.cat(quantized_slots, dim=-1)
        c_k = torch.cat(codebook_slots, dim=-1)
        k_idx = torch.stack(slot_indices, dim=1)

        return comm_out, {
            'z': z,
            'z_slots': z_slots,
            'c_k': c_k,
            'c_k_slots': codebook_slots,
            'k_idx': k_idx,
            'slot_dims': self.slot_dims,
            'slot_names': self.slot_names,
        }


class AENetwork(A3CTemplate):
    """
    An network with AE comm.
    """
    def __init__(self, obs_space, act_space, num_agents, comm_len,
                 discrete_comm, ae_pg=0, ae_type='', hidden_size=256,
                 img_feat_dim=64, method='vocal'):
        super().__init__()

        # assume action space is a Tuple of 2 spaces
        self.env_action_size = act_space[0].n  # Discrete
        self.action_size = self.env_action_size
        self.ae_pg = ae_pg
        self.method = method

        self.num_agents = num_agents

        self.comm_ae = EncoderDecoder(obs_space, comm_len, discrete_comm,
                                      num_agents, ae_type=ae_type,
                                      img_feat_dim=img_feat_dim,
                                      method=method)

        if method == 'ae-comm' and ae_type == '':
            comm_feat_len = self.comm_ae.preprocessor.feat_dim
        else:
            comm_feat_len = comm_len

        self.input_processor = InputProcessor(
            obs_space,
            comm_feat_len,
            num_agents,
            last_fc_dim=img_feat_dim)

        # individual memories
        self.feat_dim = self.input_processor.feat_dim + comm_len
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size, num_layers=1
                      ) for _ in range(num_agents)])
        self.is_recurrent = True

        # separate AC for env action and comm action
        self.env_critic_linear = nn.ModuleList([nn.Linear(
            hidden_size, 1) for _ in range(num_agents)])
        self.env_actor_linear = nn.ModuleList([nn.Linear(
            hidden_size, self.env_action_size) for _ in range(num_agents)])

        self.reset_parameters()
        return

    def reset_parameters(self):
        for m in self.env_actor_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 0.01)
            m.bias.data.fill_(0)

        for m in self.env_critic_linear:
            m.weight.data = normalized_columns_initializer(
                m.weight.data, 1.0)
            m.bias.data.fill_(0)
        return

    def init_hidden(self):
        return [head.init_hidden() for head in self.head]

    def take_action(self, policy_logit, comm_out):
        act_dict = {}
        act_logp_dict = {}
        ent_list = []
        all_act_dict = {}
        for agent_name, logits in policy_logit.items():
            act, act_logp, ent = super(AENetwork, self).take_action(logits)

            act_dict[agent_name] = act
            act_logp_dict[agent_name] = act_logp
            ent_list.append(ent)

            comm_act = (comm_out[int(agent_name[-1])]).cpu().numpy()
            all_act_dict[agent_name] = [act, comm_act]
        return act_dict, act_logp_dict, ent_list, all_act_dict

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1  # agents + global

        # WARNING: the following code only works for Python 3.6 and beyond

        # (1) pre-process inputs
        comm_feat = []
        for i in range(self.num_agents):
            cf = self.comm_ae.decode(inputs[f'agent_{i}']['comm'][:-1])
            if not self.ae_pg:
                cf = cf.detach()
            comm_feat.append(cf)

        cat_feat = self.input_processor(inputs, comm_feat)

        # (2) generate communication output and auxiliary bookkeeping
        with torch.no_grad():
            x = self.input_processor(inputs)
        x = torch.cat(x, dim=0)
        comm_out, aux_info = self.comm_ae(x)

        # (3) predict policy and values separately
        env_actor_out, env_critic_out = {}, {}

        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == 'global':
                continue

            cat_feat[i] = torch.cat([cat_feat[i], comm_out[i].unsqueeze(0)],
                                    dim=-1)

            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i])

            env_actor_out[agent_name] = self.env_actor_linear[i](x)
            env_critic_out[agent_name] = self.env_critic_linear[i](x)

            # mask logits of unavailable actions if provided
            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state, \
               comm_out.detach(), aux_info
