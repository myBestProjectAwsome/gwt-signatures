import torch
from torch import nn
from torch.nn import functional as F


class LatentPredictor(nn.Module):
    """(z_t, action) -> prédiction de z_{t+1}, entièrement dans l'espace latent.

    C'est le coeur d'un JEPA : on ne prédit pas les pixels de l'image suivante,
    seulement sa représentation abstraite. Prédiction résiduelle : le réseau
    apprend le CHANGEMENT (z_{t+1} - z_t), plus facile que l'état complet.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_actions = cfg.n_actions
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.n_actions, cfg.predictor_hidden), nn.ReLU(),
            nn.Linear(cfg.predictor_hidden, cfg.predictor_hidden), nn.ReLU(),
            nn.Linear(cfg.predictor_hidden, cfg.latent_dim),
        )

    def forward(self, z, action):
        a = F.one_hot(action, self.n_actions).float()
        return z + self.net(torch.cat([z, a], dim=-1))
