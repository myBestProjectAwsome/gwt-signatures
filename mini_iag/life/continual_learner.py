"""Apprentissage continu : l'imagination se corrige avec ce que l'agent vit.

Ce qui apprend : le PRÉDICTEUR du modèle du monde (l'imagination).
Ce qui reste figé :
  - l'encodeur (la perception) : s'il changeait, le module de coût, qui lit ses
    vecteurs, ne les comprendrait plus ; la mémoire des lieux visités non plus ;
  - le module de coût (les valeurs et l'évaluation) : il sera verrouillé à
    l'étape 5, l'agent ne doit jamais pouvoir le modifier.

Perte, sur des segments de H pas imaginés d'affilée :
  - l'état imaginé doit coller à l'état réellement atteint (comme à l'étape 2) ;
  - le module de coût (figé) doit lire dans l'état imaginé les événements qui
    ont réellement eu lieu (lave, clé, porte...). Le prédicteur apprend donc à
    produire une imagination que l'évaluateur interprète correctement.
"""
import torch
from torch.nn import functional as F


class ContinualLearner:
    def __init__(self, arch, buffer, lr=3e-4, batch_size=256, steps_per_update=40,
                 old_fraction=0.5, event_weight=1.0):
        self.arch, self.buffer = arch, buffer
        self.wm, self.cost = arch.world_model, arch.cost
        self.batch_size, self.steps, self.old_fraction = batch_size, steps_per_update, old_fraction
        self.event_weight = event_weight
        self.params = list(self.wm.predictor.parameters())
        for p in self.params:
            p.requires_grad = True
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.n_updates = 0

    def loss(self, obs, actions, visited, events, alive):
        B, H = actions.shape
        with torch.no_grad():
            z0 = self.wm.encode(obs)
            target = self.wm.target_encoder(visited.flatten(0, 1)).view(B, H, -1)
        traj = self.wm.rollout(z0, actions)
        prev = torch.cat([z0.unsqueeze(1), traj[:, :-1]], dim=1)
        m = alive.float()
        mse = ((traj - target) ** 2).mean(-1)
        x = torch.cat([prev, traj], -1) if getattr(self.cost, "pair_input", False) else traj
        logits = self.cost.body(x.flatten(0, 1)).view(B, H, -1)
        bce = F.binary_cross_entropy_with_logits(logits, events, reduction="none").mean(-1)
        n = m.sum().clamp_min(1)
        return ((mse * m).sum() / n) + self.event_weight * ((bce * m).sum() / n)

    def update(self, steps=None):
        """Une séance d'apprentissage. Renvoie la perte moyenne."""
        cost_frozen = [p.requires_grad for p in self.cost.parameters()]
        for p in self.cost.parameters():            # le coût n'est JAMAIS modifié ici
            p.requires_grad = False
        self.wm.predictor.train()
        total = 0.0
        for _ in range(steps or self.steps):
            loss = self.loss(*self.buffer.batch(self.batch_size, self.old_fraction))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total += loss.item()
        self.wm.predictor.eval()
        for p, rg in zip(self.cost.parameters(), cost_frozen):
            p.requires_grad = rg
        self.n_updates += 1
        return total / (steps or self.steps)

    @torch.no_grad()
    def surprise(self, traj):
        """Surprise moyenne d'un épisode : écart entre imagination et réalité à 1 pas."""
        obs = torch.as_tensor(traj["obs"]).float()
        nxt = torch.as_tensor(traj["next_obs"]).float()
        act = torch.as_tensor(traj["action"])
        pred = self.wm.predict(self.wm.encode(obs), act)
        return float(((pred - self.wm.target_encoder(nxt)) ** 2).sum(-1).mean())
