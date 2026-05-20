import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize_gaussian(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VQPrototypeLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, alpha=0.25, entropy_weight=0.02):
        super().__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.entropy_weight = entropy_weight
        self.prototypes = nn.Parameter(torch.empty(num_protos, latent_dim))
        self.prototypes.data.uniform_(-1.0 / num_protos, 1.0 / num_protos)

    def forward(self, latents):
        dists_to_protos = (
            torch.sum(latents ** 2, dim=1, keepdim=True)
            + torch.sum(self.prototypes ** 2, dim=1)
            - 2 * torch.matmul(latents, self.prototypes.t())
        )
        closest_protos = torch.argmin(dists_to_protos, dim=1, keepdim=True)
        encoding_one_hot = torch.zeros(
            closest_protos.size(0), self.num_protos, device=latents.device
        )
        encoding_one_hot.scatter_(1, closest_protos, 1.0)
        quantized_latents = torch.matmul(encoding_one_hot, self.prototypes)

        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        entropy_loss = self.get_categorical_entropy(dists_to_protos)
        vq_loss = (
            self.alpha * commitment_loss
            + embedding_loss
            + self.entropy_weight * entropy_loss
        )

        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, {
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
            "embedding_loss": embedding_loss,
            "entropy_loss": entropy_loss,
            "proto_indices": closest_protos.squeeze(1),
        }

    def get_categorical_entropy(self, distances):
        logdist = torch.log_softmax(-distances, dim=1)
        soft_dist = torch.mean(logdist.exp(), dim=0)
        soft_dist = soft_dist + 1e-6
        soft_dist = soft_dist / torch.sum(soft_dist)
        return torch.sum(-soft_dist * soft_dist.log())
