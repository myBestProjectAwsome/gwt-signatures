"""Modèle du monde de type JEPA (Joint-Embedding Predictive Architecture).

Principe (LeCun 2022 ; Assran et al. 2023, I-JEPA) :
    z_t      = encodeur(obs_t)                  (encodeur "en ligne", entraîné)
    ẑ_{t+1}  = prédicteur(z_t, action)
    z*_{t+1} = encodeur_cible(obs_{t+1})        (copie lente, PAS entraînée par gradient)
    perte    = || ẑ_{t+1} - z*_{t+1} ||²

Piège classique : l'effondrement. Si l'encodeur envoie toutes les observations
sur le même vecteur, la prédiction est parfaite et ne sert à rien. Deux
parades, combinées ici :
  - l'encodeur cible est une moyenne mobile (EMA) de l'encodeur en ligne ;
  - une régularisation VICReg (Bardes et al. 2022) force chaque dimension
    latente à varier (variance) et à porter une information différente
    des autres (covariance).
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .state_encoder import StateEncoder
from .latent_predictor import LatentPredictor


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = StateEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = LatentPredictor(cfg)
        self.ema_decay = cfg.ema_decay

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, z, action):
        """"Simule ce choix" : état latent suivant si on fait `action`."""
        return self.predictor(z, action)

    def rollout(self, z, actions):
        """Imagine une séquence d'actions (B, T) sans toucher au vrai monde."""
        traj = []
        for t in range(actions.shape[1]):
            z = self.predict(z, actions[:, t])
            traj.append(z)
        return torch.stack(traj, dim=1)

    # ------------------------------------------------------------ apprentissage
    def loss(self, obs, action, next_obs, var_weight=1.0, cov_weight=0.04):
        z = self.encoder(obs)
        pred = self.predictor(z, action)
        with torch.no_grad():
            target = self.target_encoder(next_obs)
        pred_loss = F.mse_loss(pred, target)
        return pred_loss + var_weight * _variance(z) + cov_weight * _covariance(z), pred_loss

    @torch.no_grad()
    def update_target(self):
        """L'encodeur cible suit lentement l'encodeur en ligne."""
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(self.ema_decay).add_(po, alpha=1 - self.ema_decay)


def _variance(z, eps=1e-4):
    """Pénalise les dimensions dont l'écart-type descend sous 1 (anti-effondrement)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1 - std).mean()


def _covariance(z):
    """Pénalise les corrélations entre dimensions (chacune doit porter sa propre info)."""
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / z.shape[1]
