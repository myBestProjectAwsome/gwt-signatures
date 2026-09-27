"""Apprentissage continu : l'imagination se corrige avec ce que l'agent vit.

Ce qui apprend : le PRÉDICTEUR du modèle du monde (l'imagination).
Ce qui reste figé :
  - l'encodeur (la perception) : s'il changeait, le module de coût, qui lit ses
    vecteurs, ne les comprendrait plus ; la mémoire des lieux visités non plus ;
  - le module de coût (les valeurs et l'évaluation) : il sera verrouillé à
    l'étape 5, l'agent ne doit jamais pouvoir le modifier.

Option learn_critic : la critique d'actions apprend aussi (critic_learner.py).

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
                 old_fraction=0.5, event_weight=1.0, learn_critic=False):
        self.arch, self.buffer = arch, buffer
        self.wm, self.cost = arch.world_model, arch.cost
        self.batch_size, self.steps, self.old_fraction = batch_size, steps_per_update, old_fraction
        self.event_weight = event_weight
        self.params = list(self.wm.predictor.parameters())
        for p in self.params:
            p.requires_grad = True
        self.senses = getattr(self.wm, "new_senses", None)
        if self.senses is not None:
            # un sens ajouté en vivant (ex. la glace) : SEULS ses poids apprennent,
            # le reste de la perception reste figé (le module de coût la lit toujours pareil)
            w = self.wm.encoder.net[0].weight
            w.requires_grad = True
            mask = torch.zeros_like(w)
            mask[:, self.senses] = 1
            w.register_hook(lambda g: g * mask)
            self.params.append(w)
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self.n_updates = 0
        self.critic_learner = None
        if learn_critic and getattr(arch, "critic", None) is not None:
            from .critic_learner import CriticLearner
            self.critic_learner = CriticLearner(arch.critic, self.wm, buffer,
                                                old_fraction=old_fraction)

    def loss(self, obs, actions, visited, events, alive):
        B, H = actions.shape
        with torch.no_grad():
            target = self.wm.target_encoder(visited.flatten(0, 1)).view(B, H, -1)
        if self.senses is not None:
            z0 = self.wm.encode(obs)                   # le gradient atteint les nouveaux sens
        else:
            with torch.no_grad():
                z0 = self.wm.encode(obs)
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
            if self.senses is not None:
                self.wm.update_new_senses()
            total += loss.item()
        self.wm.predictor.eval()
        for p, rg in zip(self.cost.parameters(), cost_frozen):
            p.requires_grad = rg
        if self.critic_learner is not None:        # la critique apprend aussi en vivant
            self.critic_learner.update()
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
