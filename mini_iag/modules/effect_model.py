"""Modèle des conséquences (feuille de route v2, lot 3) : « si j'arrive sur cette case,
à quoi ressemblera le monde ? »

C'est un modèle du monde appris, mais d'un autre grain que le JEPA : il ne prédit
pas le pas suivant d'une action, il prédit l'OBSERVATION après être entré sur une
case donnée. Il apprend sur l'expérience réelle (le marcheur au hasard) ce que
change chaque case quand on y entre : l'agent est dessus, et parfois autre chose
change aussi (un objet disparaît, un canal s'allume). Personne ne lui dit que
« la clé ouvre la porte » : il voit seulement, dans ses données, ce qui change.

Il sert à la planification par événements (event_planner.py) : pour savoir si
aller d'abord ailleurs change ce qui devient atteignable ensuite.

Entrée : l'observation (C canaux) + un masque de la case visée.
Sortie : l'observation prédite après l'arrivée (probabilité de chaque case/canal).
Une branche globale (maximum sur la carte) permet de prédire des changements qui
concernent toute la carte, comme le canal « clé en main ».
"""
import torch
from torch import nn

from ..environment.keydoor_world import N_CHANNELS


class EffectModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        c = N_CHANNELS + 1
        self.local = nn.Sequential(nn.Conv2d(c, hidden, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(2 * hidden + c, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, N_CHANNELS, 1))

    def forward(self, obs, cell_mask):
        """obs (B, C, H, W), cell_mask (B, H, W) -> logits de l'observation suivante (B, C, H, W)."""
        x = torch.cat([obs[:, :N_CHANNELS].float(), cell_mask[:, None].float()], 1)
        h = self.local(x)
        g = h.amax((-1, -2), keepdim=True).expand_as(h)
        return self.out(torch.cat([h, g, x], 1))

    @torch.no_grad()
    def imagine(self, obs, cell_mask):
        """Observation prédite (0/1) après l'arrivée sur la case."""
        return (torch.sigmoid(self(obs, cell_mask)) > 0.5).float()
