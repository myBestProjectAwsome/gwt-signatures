"""Carte mentale : une carte des valeurs sur TOUTES les cases, pour planifier loin.

Le problème (diagnostics 8 et 10) : l'imagination voit 5 pas devant, la critique
d'actions (un réseau qui lit le vecteur latent) devine mal au-delà de 7 pas.
Or trouver un chemin de 20 pas dans un labyrinthe demande de PROPAGER
l'information de proche en proche, ce qu'un réseau à une seule passe fait mal.

La carte mentale fait cette propagation. Pour chaque événement visé (objectif,
clé, porte), elle calcule une valeur sur chaque case de la carte :
  1. ce qu'il y a sur chaque case, appris : la RÉCOMPENSE (l'événement visé
     arrive-t-il si j'y entre ?), le BLOCAGE (est-ce que je reste sur place ?),
     le DANGER (est-ce que j'y meurs ?) ;
  2. où mène chaque action, appris : un petit noyau 3x3 par action (au départ,
     l'agent ne sait pas que « haut » mène à la case du dessus) ;
  3. la valeur se propage de case en case pendant K itérations :
        valeur(case) = (1 - blocage)(1 - danger) x [récompense
                       + (1 - récompense) x discount x meilleure valeur voisine]
     C'est l'itération sur les valeurs (Bellman), faite par convolution : l'idée
     des Value Iteration Networks (Tamar et al., 2016), et une forme simple de
     « carte cognitive » (Tolman, 1948).
  4. On lit la valeur de chaque action à la position de l'agent.

Tout est appris par Q-learning hors ligne sur l'expérience réelle, comme la
critique d'actions (map_trainer.py). Rien n'est codé à la main : ni où sont les
murs, ni ce que fait une porte, ni la direction des actions. Ce qui est donné,
c'est la FORME du calcul : un monde en cases, où l'on se déplace de proche en
proche. C'est un a priori, affiché comme tel.

Elle lit l'observation (la grille), pas le vecteur latent.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..environment.keydoor_world import AGENT, N_CHANNELS

TARGETS = ("goal", "key", "door")


class MentalMap(nn.Module):
    uses_obs = True

    def __init__(self, cfg, hidden=32, iterations=30, discount=0.9):
        super().__init__()
        self.iterations, self.discount = iterations, discount
        self.features = nn.Sequential(nn.Conv2d(N_CHANNELS, hidden, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(hidden, hidden, 1), nn.ReLU())
        self.reward = nn.Conv2d(hidden, len(TARGETS), 1)
        self.blocked = nn.Conv2d(hidden, 1, 1)
        self.deadly = nn.Conv2d(hidden, 1, 1)
        self.moves = nn.Parameter(0.01 * torch.randn(cfg.n_actions, 9))   # où mène chaque action

    def kernels(self):
        return torch.softmax(self.moves, -1).view(-1, 1, 3, 3)

    def _shift(self, U, K):
        """(B, T, H, W) -> (B, T, A, H, W) : valeur de la case où mène chaque action."""
        B, T, H, W = U.shape
        return F.conv2d(U.reshape(B * T, 1, H, W), K, padding=1).view(B, T, -1, H, W)

    def maps(self, obs):
        """Cartes apprises : récompense (B, 3, H, W), blocage et danger (B, 1, H, W)."""
        h = self.features(obs[:, :N_CHANNELS].float())
        return torch.sigmoid(self.reward(h)), torch.sigmoid(self.blocked(h)), torch.sigmoid(self.deadly(h))

    def values(self, obs):
        """Valeur d'ARRIVER sur chaque case, pour chaque événement visé : (B, 3, H, W)."""
        r, b, d = self.maps(obs)
        K, free = self.kernels(), (1 - b) * (1 - d)
        U = free * r
        for _ in range(self.iterations):
            U = free * (r + (1 - r) * self.discount * self._shift(U, K).max(2).values)
        return U

    def forward(self, obs):
        """(B, C, H, W) -> (B, 3, 4) : valeur de chaque action, pour chaque événement visé."""
        obs = obs.float()
        U = self.values(obs)
        _, b, _ = self.maps(obs)
        K = self.kernels()
        here = obs[:, AGENT][:, None, None]                                   # (B, 1, 1, H, W)
        arrive = (self._shift(U, K) * here).sum((-1, -2))                      # (B, 3, A)
        stuck = (self._shift(b, K) * here).sum((-1, -2))                       # (B, 1, A)
        best = arrive.max(-1, keepdim=True).values
        # action bloquée : on reste sur place, on a perdu un pas
        return arrive + stuck * self.discount * best

    def q(self, obs, target):
        return self(obs)[:, TARGETS.index(target)]
