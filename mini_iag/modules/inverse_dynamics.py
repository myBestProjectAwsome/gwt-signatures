import torch
from torch import nn


class InverseDynamics(nn.Module):
    """(z_t, z_{t+1}) -> quelle action a été faite ? (logits sur les 4 actions)

    Tête auxiliaire du modèle du monde. Sans elle, le JEPA apprend à ignorer
    l'agent : la carte ne bouge pas d'un pas à l'autre, donc un latent qui
    n'encode QUE la carte se prédit parfaitement lui-même (cf. README, étape 2).
    Pour deviner l'action, le latent doit encoder ce que l'agent CONTRÔLE.
    Reste auto-supervisé : l'agent connaît toujours l'action qu'il a faite.
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.latent_dim, cfg.predictor_hidden), nn.ReLU(),
            nn.Linear(cfg.predictor_hidden, cfg.n_actions),
        )

    def forward(self, z, z_next):
        return self.net(torch.cat([z, z_next], dim=-1))
