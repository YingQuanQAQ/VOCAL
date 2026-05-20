from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from bootstrap import ensure_shared_training_path, get_repo_root

ensure_shared_training_path()

from model.a3c_template import A3CTemplate  # noqa: E402
from model.init import normalized_columns_initializer  # noqa: E402
from model.model_utils import ImgModule, LSTMhead  # noqa: E402
from vqvib_utils import VQPrototypeLayer, reparameterize_gaussian  # noqa: E402


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        del ctx
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        del ctx
        return F.hardtanh(grad_output)


OV_SLOT_NAMES = ["action_intent", "entity", "location_status"]
OV_SLOT_VOCABS = [
    [
        "I am picking up",
        "I am putting down",
        "I am waiting for",
        "Please get",
    ],
    [
        "an Onion",
        "a Plate",
        "the Soup",
        "Nothing / Empty hands",
    ],
    [
        "from the Dispenser",
        "to the Pot",
        "to the Serving Station",
        "at the Counter",
    ],
]


class InputProcessor(nn.Module):
    def __init__(self, obs_space, comm_feat_len, num_agents, last_fc_dim=64):
        super(InputProcessor, self).__init__()

        self.obs_keys = list(obs_space.spaces.keys())
        self.num_agents = num_agents
        self.conv = ImgModule(obs_space["pov"].shape, last_fc_dim=last_fc_dim)
        feat_dim = last_fc_dim

        state_feat_dim = 0
        if "self_env_act" in self.obs_keys:
            self.env_act_dim = obs_space.spaces["self_env_act"].n
            state_feat_dim += self.env_act_dim

        if "selfpos" in self.obs_keys:
            self.discrete_positions = None
            if obs_space.spaces["selfpos"].__class__.__name__ == "MultiDiscrete":
                self.discrete_positions = obs_space.spaces["selfpos"].nvec
                state_feat_dim += sum(self.discrete_positions)
            else:
                state_feat_dim += 2

        if state_feat_dim == 0:
            self.state_feat_fc = None
        else:
            self.state_feat_fc = nn.Linear(state_feat_dim, 64)
            feat_dim += 64

        if self.state_feat_fc is not None:
            self.state_layer_norm = nn.LayerNorm(64)
        self.img_layer_norm = nn.LayerNorm(last_fc_dim)

        self.comm_feat_dim = comm_feat_len * (num_agents - 1)
        feat_dim += self.comm_feat_dim
        self.feat_dim = feat_dim

    def forward(self, inputs, comm=None):
        pov = []
        for i in range(self.num_agents):
            pov.append(inputs["agent_{}".format(i)]["pov"])
        x = torch.cat(pov, dim=0)
        if x.shape[-2] < 20 or x.shape[-1] < 20:
            x = F.interpolate(x, size=(20, 20), mode="nearest")
        x = self.conv(x)
        xs = torch.chunk(x, self.num_agents)

        cat_feat = [self.img_layer_norm(xs[i]) for i in range(self.num_agents)]

        for i in range(self.num_agents):
            if self.state_feat_fc is not None:
                feats = []
                if "self_env_act" in self.obs_keys:
                    env_act = F.one_hot(
                        inputs["agent_{}".format(i)]["self_env_act"].to(torch.int64),
                        num_classes=self.env_act_dim,
                    )
                    feats.append(torch.reshape(env_act, (1, self.env_act_dim)))

                if "selfpos" in self.obs_keys:
                    sp = inputs["agent_{}".format(i)]["selfpos"].to(torch.int64)
                    if self.discrete_positions is not None:
                        spx = F.one_hot(
                            sp[0], num_classes=self.discrete_positions[0]
                        )
                        spy = F.one_hot(
                            sp[1], num_classes=self.discrete_positions[1]
                        )
                        sp = torch.cat([spx, spy], dim=-1).float()
                        sp = torch.reshape(sp, (1, sum(self.discrete_positions)))
                    else:
                        sp = torch.reshape(sp, (1, 2))
                    feats.append(sp)

                feats = torch.cat(feats, dim=-1)
                feats = self.state_feat_fc(feats)
                feats = self.state_layer_norm(feats)
                cat_feat[i] = torch.cat([cat_feat[i], feats], dim=-1)

            if comm is not None:
                c = torch.reshape(comm[i], (1, self.comm_feat_dim))
                cat_feat[i] = torch.cat([cat_feat[i], c], dim=-1)

        return cat_feat


class EncoderDecoder(nn.Module):
    def __init__(self, obs_space, comm_len, discrete_comm, num_agents,
                 ae_type="", img_feat_dim=64, method="vocal",
                 vqvib_num_protos=16, vqvib_beta=0.05,
                 vqvib_commit_alpha=0.25, vqvib_entropy_weight=0.02,
                 vqvib_recons_weight=1.0, slot_grouping="task"):
        super(EncoderDecoder, self).__init__()

        self.preprocessor = InputProcessor(
            obs_space, 0, num_agents, last_fc_dim=img_feat_dim
        )
        in_size = self.preprocessor.feat_dim

        self.method = method
        self.discrete_comm = discrete_comm
        self.ae_type = ae_type
        self.comm_len = comm_len
        self.vqvib_beta = vqvib_beta
        self.vqvib_recons_weight = vqvib_recons_weight
        self.slot_grouping = slot_grouping
        self.base_slot_names = None
        self.base_slot_vocabs = None
        self.base_slot_dims = None

        if self.method == "ae-comm":
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, comm_len),
                nn.Sigmoid(),
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
        elif self.method == "vocal":
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, comm_len),
            )
            self.slot_names = list(OV_SLOT_NAMES)
            self.slot_vocabs = [list(vocab) for vocab in OV_SLOT_VOCABS]
            self.slot_dims, slot_embeddings = self.load_slot_dictionary(
                "text_embeddings_overcooked.pt"
            )
            self.slot_names, self.slot_vocabs, self.slot_dims, slot_embeddings = \
                self.apply_slot_grouping(
                    self.slot_names, self.slot_vocabs,
                    self.slot_dims, slot_embeddings)
            self.num_slots = len(self.slot_dims)
            self.codebooks = nn.ParameterList()
            for slot_idx, embeddings in enumerate(slot_embeddings):
                self.register_buffer("text_anchor_{}".format(slot_idx), embeddings)
                self.codebooks.append(nn.Parameter(embeddings.clone()))
        elif self.method == "vqvib":
            self.encoder = nn.Sequential(
                nn.Linear(in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, comm_len),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(comm_len, comm_len)
            self.fc_var = nn.Linear(comm_len, comm_len)
            self.vq_layer = VQPrototypeLayer(
                num_protos=vqvib_num_protos,
                latent_dim=comm_len,
                alpha=vqvib_commit_alpha,
                entropy_weight=vqvib_entropy_weight,
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
            raise ValueError("Unknown method: {}".format(self.method))

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

    def build_single_slot_dictionary(self, slot_names, slot_vocabs, slot_dims,
                                     slot_embeddings):
        joint_name = ["__".join(slot_names)]
        joint_vocab = []
        joint_embeddings = []
        ranges = [range(len(vocab)) for vocab in slot_vocabs]
        for token_indices in itertools.product(*ranges):
            joint_vocab.append(" | ".join(
                slot_vocabs[slot_idx][token_idx]
                for slot_idx, token_idx in enumerate(token_indices)
            ))
            joint_embeddings.append(torch.cat(
                [slot_embeddings[slot_idx][token_idx]
                 for slot_idx, token_idx in enumerate(token_indices)],
                dim=0,
            ))
        joint_embeddings = [torch.stack(joint_embeddings, dim=0)]
        return joint_name, [joint_vocab], [sum(slot_dims)], joint_embeddings

    def apply_slot_grouping(self, slot_names, slot_vocabs, slot_dims,
                            slot_embeddings):
        self.base_slot_names = list(slot_names)
        self.base_slot_vocabs = [list(vocab) for vocab in slot_vocabs]
        self.base_slot_dims = list(slot_dims)
        if self.slot_grouping == "single":
            return self.build_single_slot_dictionary(
                self.base_slot_names, self.base_slot_vocabs,
                self.base_slot_dims, slot_embeddings)
        return slot_names, slot_vocabs, slot_dims, slot_embeddings

    def format_slot_targets(self, slot_target):
        if self.slot_grouping != "single" or slot_target is None:
            return slot_target
        flat_idx = 0
        for target_idx, vocab in zip(slot_target, self.base_slot_vocabs):
            flat_idx = flat_idx * len(vocab) + target_idx
        return [flat_idx]

    def load_slot_dictionary(self, file_name):
        anchor_path = os.path.join(get_repo_root(), file_name)
        fallback_slot_dims = self.get_slot_dims(self.comm_len, len(self.slot_vocabs))

        try:
            payload = torch.load(anchor_path, map_location="cpu")
            slot_dims = payload["slot_dims"]
            slot_embeddings = payload["slot_embeddings"]
            slot_names = payload.get("slot_names")
            slot_vocabs = payload.get("slot_vocabs")

            if sum(slot_dims) != self.comm_len:
                raise ValueError("slot dims do not match comm_len")
            if len(slot_embeddings) != len(slot_dims):
                raise ValueError("slot embedding count does not match slot dims")
            if slot_names is not None:
                self.slot_names = slot_names
            if slot_vocabs is not None:
                self.slot_vocabs = slot_vocabs
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            print(
                "Warning: {} invalid or not found at {}, initializing random slot anchors.".format(
                    file_name, anchor_path
                )
            )
            slot_dims = fallback_slot_dims
            slot_embeddings = self.build_random_slot_embeddings(
                self.slot_vocabs, slot_dims
            )

        return slot_dims, slot_embeddings

    def get_text_anchors(self):
        return [
            getattr(self, "text_anchor_{}".format(idx))
            for idx in range(len(self.codebooks))
        ]

    def get_anchor_loss(self):
        losses = []
        for codebook, anchor in zip(self.codebooks, self.get_text_anchors()):
            losses.append(F.mse_loss(codebook, anchor))
        return torch.stack(losses).mean()

    def get_anchor_distance(self):
        distances = []
        for codebook, anchor in zip(self.codebooks, self.get_text_anchors()):
            distances.append(torch.norm(codebook - anchor, dim=1).mean())
        return torch.stack(distances).mean()

    def decode(self, x):
        if self.method in {"vocal", "vqvib"}:
            return x
        return self.decoder(x)

    def forward(self, feat):
        if self.method == "ae-comm":
            encoded = self.encoder(feat)
            if self.discrete_comm:
                encoded = STE.apply(encoded)
            decoded = self.decoder(encoded)
            loss = F.mse_loss(decoded, feat)
            return encoded.detach(), {"recon_loss": loss}

        if self.method == "vqvib":
            hidden = self.encoder(feat)
            mu = self.fc_mu(hidden)
            logvar = self.fc_var(hidden)
            sample = reparameterize_gaussian(mu, logvar)
            quantized, vq_info = self.vq_layer(sample)
            decoded = self.decoder(quantized)
            recon_loss = F.mse_loss(decoded, feat)
            kl_loss = torch.mean(
                -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1),
                dim=0,
            )
            total_loss = (
                self.vqvib_beta * kl_loss
                + vq_info["vq_loss"]
                + self.vqvib_recons_weight * recon_loss
            )
            return quantized, {
                "vq_total_loss": total_loss,
                "vq_loss": vq_info["vq_loss"],
                "vq_commitment_loss": vq_info["commitment_loss"],
                "vq_embedding_loss": vq_info["embedding_loss"],
                "vq_entropy_loss": vq_info["entropy_loss"],
                "vq_kl_loss": kl_loss,
                "vq_recon_loss": recon_loss,
                "proto_indices": vq_info["proto_indices"],
                "mu": mu,
                "logvar": logvar,
            }

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
            "z": z,
            "z_slots": z_slots,
            "c_k": c_k,
            "c_k_slots": codebook_slots,
            "k_idx": k_idx,
            "slot_dims": self.slot_dims,
            "slot_names": self.slot_names,
        }


class AENetwork(A3CTemplate):
    def __init__(self, obs_space, act_space, num_agents, comm_len,
                 discrete_comm, ae_pg=0, ae_type="", hidden_size=256,
                 img_feat_dim=64, method="vocal",
                 vqvib_num_protos=16, vqvib_beta=0.05,
                 vqvib_commit_alpha=0.25, vqvib_entropy_weight=0.02,
                 vqvib_recons_weight=1.0, slot_grouping="task"):
        super().__init__()

        self.env_action_size = act_space[0].n
        self.action_size = self.env_action_size
        self.ae_pg = ae_pg
        self.method = method
        self.num_agents = num_agents

        self.comm_ae = EncoderDecoder(
            obs_space,
            comm_len,
            discrete_comm,
            num_agents,
            ae_type=ae_type,
            img_feat_dim=img_feat_dim,
            method=method,
            vqvib_num_protos=vqvib_num_protos,
            vqvib_beta=vqvib_beta,
            vqvib_commit_alpha=vqvib_commit_alpha,
            vqvib_entropy_weight=vqvib_entropy_weight,
            vqvib_recons_weight=vqvib_recons_weight,
            slot_grouping=slot_grouping,
        )

        comm_feat_len = self.comm_ae.preprocessor.feat_dim \
            if method == "ae-comm" and ae_type == "" else comm_len

        self.input_processor = InputProcessor(
            obs_space,
            comm_feat_len,
            num_agents,
            last_fc_dim=img_feat_dim,
        )

        self.feat_dim = self.input_processor.feat_dim + comm_len
        self.head = nn.ModuleList(
            [LSTMhead(self.feat_dim, hidden_size, num_layers=1)
             for _ in range(num_agents)]
        )
        self.is_recurrent = True

        self.env_critic_linear = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_agents)]
        )
        self.env_actor_linear = nn.ModuleList(
            [nn.Linear(hidden_size, self.env_action_size)
             for _ in range(num_agents)]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.env_actor_linear:
            m.weight.data = normalized_columns_initializer(m.weight.data, 0.01)
            m.bias.data.fill_(0)

        for m in self.env_critic_linear:
            m.weight.data = normalized_columns_initializer(m.weight.data, 1.0)
            m.bias.data.fill_(0)

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
            comm_act = comm_out[int(agent_name[-1])].cpu().numpy()
            all_act_dict[agent_name] = [act, comm_act]
        return act_dict, act_logp_dict, ent_list, all_act_dict

    def forward(self, inputs, hidden_state=None, env_mask_idx=None):
        assert type(inputs) is dict
        assert len(inputs.keys()) == self.num_agents + 1

        comm_feat = []
        for i in range(self.num_agents):
            cf = self.comm_ae.decode(inputs["agent_{}".format(i)]["comm"][:-1])
            if not self.ae_pg:
                cf = cf.detach()
            comm_feat.append(cf)

        cat_feat = self.input_processor(inputs, comm_feat)

        with torch.no_grad():
            x = self.input_processor(inputs)
        x = torch.cat(x, dim=0)
        comm_out, aux_info = self.comm_ae(x)

        env_actor_out, env_critic_out = {}, {}
        for i, agent_name in enumerate(inputs.keys()):
            if agent_name == "global":
                continue

            cat_feat[i] = torch.cat([cat_feat[i], comm_out[i].unsqueeze(0)], dim=-1)
            x, hidden_state[i] = self.head[i](cat_feat[i], hidden_state[i])
            env_actor_out[agent_name] = self.env_actor_linear[i](x)
            env_critic_out[agent_name] = self.env_critic_linear[i](x)

            if env_mask_idx and env_mask_idx[i]:
                env_actor_out[agent_name][0, env_mask_idx[i]] = -1e10

        return env_actor_out, env_critic_out, hidden_state, comm_out.detach(), aux_info
