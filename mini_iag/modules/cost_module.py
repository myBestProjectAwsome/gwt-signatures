"""Module de coût : évalue un état latent.

Deux têtes :
  - danger  : probabilité que l'état soit dangereux (lave)
  - succès  : probabilité que l'état soit l'objectif
  coût = w_danger * danger - w_succes * succès   (bas = bon)

Garde-fou (principe du dépôt, utilisé à l'étape 5) : lock() gèle les poids
et enregistre une empreinte SHA-256 ; verify() détecte toute modification.
Le reste du système pourra lire ce module, jamais l'entraîner ni le changer.
"""
import hashlib

import torch
from torch import nn


class CostModule(nn.Module):
    def __init__(self, cfg, w_danger=1.0, w_success=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.cost_hidden), nn.ReLU(),
            nn.Linear(cfg.cost_hidden, 2),
        )
        self.w_danger, self.w_success = w_danger, w_success
        self._fingerprint = None

    def forward(self, z):
        """Renvoie (danger, succès, coût), chacun de forme (B,)."""
        logits = self.body(z)
        danger, success = torch.sigmoid(logits).unbind(-1)
        return danger, success, self.w_danger * danger - self.w_success * success

    # ------------------------------------------------------------ garde-fou
    def fingerprint(self):
        h = hashlib.sha256()
        for name, t in sorted(self.state_dict().items()):
            h.update(name.encode())
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def lock(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self._fingerprint = self.fingerprint()
        return self._fingerprint

    @property
    def locked(self):
        return self._fingerprint is not None

    def verify(self):
        """True si les poids n'ont pas bougé depuis lock()."""
        return self.locked and self.fingerprint() == self._fingerprint
