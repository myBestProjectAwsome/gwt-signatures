"""Espace de travail global : un goulot d'attention entre modules.

Inspiré de Goyal et al. (2022), "Coordination Among Neural Modules Through
a Shared Global Workspace" :

  1. ÉCRITURE (compétition) : K "slots" appris lisent les N contenus proposés
     par les modules, via une attention. Comme K << N, les contenus sont en
     compétition pour entrer : c'est le goulot.
  2. DIFFUSION (broadcast) : chaque contenu relit ensuite les K slots. Tous
     les modules reçoivent donc la même information sélectionnée.

C'est la version apprenable du workspace de `psyche` : au lieu d'une règle
"le plus saillant gagne", la sélection est faite par des poids entraînables.
"""
import torch
from torch import nn


class BottleneckWorkspace(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.latent_dim
        self.slots = nn.Parameter(torch.randn(cfg.workspace_slots, d) * 0.02)
        self.write = nn.MultiheadAttention(d, cfg.workspace_heads, batch_first=True)
        self.read = nn.MultiheadAttention(d, cfg.workspace_heads, batch_first=True)
        self.norm_slots = nn.LayerNorm(d)
        self.norm_tokens = nn.LayerNorm(d)

    def forward(self, tokens):
        """tokens : (B, N, D) contenus proposés par les modules.

        Renvoie :
          slots      (B, K, D) : contenu admis dans le workspace
          broadcast  (B, N, D) : contenus mis à jour par la diffusion
          attention  (B, K, N) : qui a gagné l'accès (somme = 1 sur N)
        """
        b = tokens.shape[0]
        queries = self.slots.unsqueeze(0).expand(b, -1, -1)
        written, attention = self.write(queries, tokens, tokens,
                                        need_weights=True, average_attn_weights=True)
        slots = self.norm_slots(queries + written)
        read, _ = self.read(tokens, slots, slots, need_weights=False)
        broadcast = self.norm_tokens(tokens + read)
        return slots, broadcast, attention
