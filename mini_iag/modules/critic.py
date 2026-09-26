"""Critique (valeur) : « à quel point suis-je proche de l'événement visé ? »

La partie « critique entraînable » du module de coût chez LeCun (2022), et la
valeur terminale de TD-MPC. Pour chaque état latent, une valeur par événement
visé possible (objectif, clé, porte), entre 0 et 1 : environ discount^distance.

Pourquoi : le planificateur n'imagine que H pas. Au-delà, sans critique,
il n'a aucun indice sur la direction à prendre (mesuré : 0 à 13 % de succès
quand la cible est à 7 pas ou plus).

Elle ne fait pas partie des « valeurs » verrouillées : c'est une estimation
apprise, pas une préférence. Les préférences (w_danger, w_succès) restent
dans le module de coût.
"""
import torch
from torch import nn

TARGETS = ("goal", "key", "door")


class Critic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.predictor_hidden), nn.ReLU(),
            nn.Linear(cfg.predictor_hidden, cfg.predictor_hidden), nn.ReLU(),
            nn.Linear(cfg.predictor_hidden, len(TARGETS)),
        )

    def forward(self, z):
        """(B, D) -> (B, 3) valeurs pour objectif, clé, porte."""
        return torch.sigmoid(self.net(z))

    def value(self, z, target):
        return self(z)[:, TARGETS.index(target)]
