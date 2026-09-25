"""Entraînement du module de coût (étape 2b), SUR les latents du modèle du monde.

Le modèle du monde est gelé : on n'entraîne que le coût. Il apprend à lire
les vecteurs que le modèle du monde produit (ils "parlent la même langue").

Étiquettes : l'agent est-il sur la lave (danger) / sur l'objectif (succès) ?
Ces états sont rares (~8 % et ~3 %) : on les repondère (pos_weight).
"""
import torch
from torch.nn import functional as F

from ..metrics import auc, labels


class CostTrainer:
    def __init__(self, cost, world_model, lr=3e-3, iters=2000, batch_size=512, seed=0):
        self.cost, self.wm = cost, world_model
        self.iters, self.batch_size = iters, batch_size
        self.opt = torch.optim.Adam(cost.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    @torch.no_grad()
    def _latents(self, obs):
        return self.wm.target_encoder(obs)

    def fit(self, train, test=None, eval_every=250, verbose=True):
        z_all = self._latents(train.next_obs)
        y_all = torch.stack(labels(train.next_obs), 1)          # (B, 2) danger, succès
        pos = y_all.mean(0).clamp_min(1e-3)
        pos_weight = (1 - pos) / pos
        if test is not None:
            z_test = self._latents(test.next_obs)
            lava_t, goal_t = labels(test.next_obs)
        self.cost.train()
        for it in range(1, self.iters + 1):
            i = torch.randint(len(z_all), (self.batch_size,), generator=self.gen)
            logits = self.cost.body(z_all[i])
            loss = F.binary_cross_entropy_with_logits(logits, y_all[i], pos_weight=pos_weight)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if it % eval_every == 0 or it == 1:
                row = {"iter": it, "loss": loss.item()}
                if test is not None:
                    with torch.no_grad():
                        d, s, _ = self.cost(z_test)
                    row.update(auc_lava=auc(d, lava_t), auc_goal=auc(s, goal_t))
                self.log.append(row)
                if verbose and test is not None:
                    print(f"  [coût] {it:5d}/{self.iters}  perte {row['loss']:.3f}  "
                          f"AUC lave {row['auc_lava']:.3f}  objectif {row['auc_goal']:.3f}", flush=True)
        self.cost.eval()
        return self.log
