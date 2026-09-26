"""Critique d'actions : « si je fais cette action ici, à quel point suis-je proche
de l'événement visé ? » — apprise sur l'expérience RÉELLE (Q-learning hors ligne).

Pour chaque état latent et chaque événement visé (objectif, clé, porte), une
valeur par action, entre 0 et 1 : environ discount^(pas restants) en jouant au
mieux ensuite. La lave vaut 0.

Pourquoi une troisième critique (cf. README, planification longue) :
  - apprise dans l'IMAGINATION, elle devenait bruitée au-delà de 5 pas ;
  - apprise sur les trajectoires d'un agent qui marche au hasard (a posteriori),
    ses valeurs s'effondraient au-delà de 3 pas (il n'atteint presque jamais une
    cible lointaine) ;
  - apprise par Q-learning hors ligne (on apprend « le mieux que j'aurais pu
    faire », pas « ce que le marcheur a fait »), elle décroît régulièrement
    jusqu'à 6 pas.
Elle n'est interrogée que sur l'état RÉEL actuel : sur des états imaginés, le
planificateur trouvait toujours les plans où elle se trompe.
"""
import torch
from torch import nn

TARGETS = ("goal", "key", "door")


class ActionCritic(nn.Module):
    def __init__(self, cfg, hidden=256):
        super().__init__()
        self.n_actions = cfg.n_actions
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(TARGETS) * cfg.n_actions),
        )

    def forward(self, z):
        """(B, D) -> (B, 3, 4) : valeur de chaque action pour chaque événement visé."""
        return torch.sigmoid(self.net(z)).view(-1, len(TARGETS), self.n_actions)

    def q(self, z, target):
        """(B, D) -> (B, 4) valeurs des actions pour `target`."""
        return self(z)[:, TARGETS.index(target)]

    def value(self, z, target):
        return self.q(z, target).max(-1).values
