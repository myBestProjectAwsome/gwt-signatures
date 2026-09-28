"""Apprendre le modèle des conséquences sur l'expérience réelle.

Chaque transition vécue (obs, obs suivante) donne un exemple : la case où
l'agent se trouve APRÈS le pas est la case « visée », et l'observation suivante
est ce qu'il faut prédire. Les cases qui changent rarement (un objet qui
disparaît, le canal clé en main qui s'allume) sont surpondérées : sinon le
modèle apprendrait « rien ne change sauf l'agent », qui est vrai 99 % du temps.
"""
import torch
from torch.nn import functional as F

from ..environment.keydoor_world import AGENT, N_CHANNELS


class EffectTrainer:
    def __init__(self, model, lr=2e-3, iters=6000, batch_size=256, change_weight=20.0, seed=0):
        self.model, self.iters, self.batch_size = model, iters, batch_size
        self.change_weight = change_weight
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    def fit(self, transitions, verbose=True, eval_every=1000):
        self.model.train()
        for it in range(1, self.iters + 1):
            i = torch.randint(len(transitions), (self.batch_size,), generator=self.gen)
            obs = transitions.obs[i].float()[:, :N_CHANNELS]
            nxt = transitions.next_obs[i].float()[:, :N_CHANNELS]
            logits = self.model(obs, nxt[:, AGENT])
            changed = (obs != nxt).float()
            changed[:, AGENT] = 0                     # le déplacement de l'agent est facile
            w = 1 + self.change_weight * changed
            loss = (F.binary_cross_entropy_with_logits(logits, nxt, reduction="none") * w).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if verbose and (it % eval_every == 0 or it == 1):
                print(f"  [conséquences] {it:5d}/{self.iters}  perte {loss.item():.5f}", flush=True)
            self.log.append(loss.item()) if it % eval_every == 0 else None
        self.model.eval()
        return self.log
