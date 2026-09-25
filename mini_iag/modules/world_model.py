"""Modèle du monde de type JEPA (Joint-Embedding Predictive Architecture).

Principe (LeCun 2022 ; Assran et al. 2023, I-JEPA) :
    z_t      = encodeur(obs_t)                  (encodeur "en ligne", entraîné)
    ẑ_{t+1}  = prédicteur(z_t, action)
    z*_{t+1} = encodeur_cible(obs_{t+1})        (copie lente, PAS entraînée par gradient)
    perte    = || ẑ_{t+1} - z*_{t+1} ||²

Piège classique : l'effondrement. Si l'encodeur envoie toutes les observations
sur le même vecteur, la prédiction est parfaite et ne sert à rien. Parades :
  - l'encodeur cible est une moyenne mobile (EMA) de l'encodeur en ligne ;
  - une régularisation VICReg (Bardes et al. 2022) force chaque dimension
    latente à varier (variance) et à porter une information différente
    des autres (covariance) ;
  - une tête de dynamique inverse (devine l'action à partir de z_t, z_{t+1}).
    Sans elle, on observe un effondrement PARTIEL : l'encodeur ne garde que la
    carte (statique, donc parfaitement prévisible, et assez variée d'un exemple
    à l'autre pour satisfaire VICReg) et jette la position de l'agent.

Et une tête d'événements (lave / objectif perçus), comme les modèles du monde
de type Dreamer qui prédisent la récompense : sans elle, l'encodeur jette la
position de l'objectif, inutile pour prédire les déplacements.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .state_encoder import StateEncoder
from .latent_predictor import LatentPredictor
from .inverse_dynamics import InverseDynamics
from .event_predictor import EventPredictor


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = StateEncoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = LatentPredictor(cfg)
        self.inverse = InverseDynamics(cfg)
        self.events = EventPredictor(cfg)
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
    def _event_loss(self, logits, events, success_weight, reduction="mean"):
        """BCE des événements ; le succès (rare) peut être surpondéré."""
        pw = torch.tensor([1.0, success_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, events, pos_weight=pw,
                                                  reduction=reduction)

    def loss(self, obs, action, next_obs, next_events=None, var_weight=1.0,
             cov_weight=0.04, inv_weight=1.0, event_weight=1.0, success_weight=1.0):
        """next_events : (B, 2) lave / objectif perçus dans next_obs (0/1).
        Renvoie (perte totale, détail des termes)."""
        z = self.encoder(obs)
        z_next = self.encoder(next_obs)
        pred = self.predictor(z, action)
        with torch.no_grad():
            target = self.target_encoder(next_obs)
        terms = {
            "prediction": F.mse_loss(pred, target),
            "variance": _variance(z),
            "covariance": _covariance(z),
            "inverse": F.cross_entropy(self.inverse(z, z_next), action),
        }
        total = (terms["prediction"] + var_weight * terms["variance"]
                 + cov_weight * terms["covariance"] + inv_weight * terms["inverse"])
        if next_events is not None and event_weight > 0:
            # sur l'état réel (forme l'encodeur) ET sur l'état imaginé (forme le prédicteur)
            terms["events"] = (self._event_loss(self.events(z_next), next_events, success_weight)
                               + self._event_loss(self.events(pred), next_events, success_weight))
            total = total + event_weight * terms["events"]
        return total, terms

    def multistep_loss(self, obs, actions, visited, events, alive, event_weight=1.0,
                       success_weight=1.0):
        """Imagine H pas d'affilée et corrige chaque pas (état ET événements).
        Sans cela, le modèle n'est entraîné qu'à un pas et dérive vite en imagination."""
        B, H = actions.shape
        with torch.no_grad():
            targets = self.target_encoder(visited.flatten(0, 1)).view(B, H, -1)
        traj = self.rollout(self.encoder(obs), actions)                # (B, H, D)
        mask = alive.float()
        mse = ((traj - targets) ** 2).mean(-1)
        bce = self._event_loss(self.events(traj), events, success_weight,
                               reduction="none").mean(-1)
        n = mask.sum().clamp_min(1)
        return ((mse + event_weight * bce) * mask).sum() / n

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
