"""Affinage de coordination (étape 3) : le module de coût apprend la "langue"
des états IMAGINÉS.

À l'étape 2, le coût n'a vu que des états réels. Or le planificateur lui
soumet des états imaginés par le modèle du monde, un peu différents. Résultat
mesuré : sur des états imaginés, ses probabilités de succès étaient faibles et
bruitées, et le planificateur choisissait des plans "hallucinés".

Ici on l'entraîne sur des états réels ET imaginés (jusqu'à H pas), étiquetés
par ce qui s'est vraiment passé. Deux différences avec l'étape 2 :
  - pas de repondération des exemples rares (pos_weight) : elle améliore le
    classement (AUC) mais GONFLE les probabilités ; or le planificateur calcule
    des espérances, il lui faut des probabilités calibrées ;
  - le modèle du monde reste gelé : seul le coût bouge.
"""
import torch
from torch.nn import functional as F


class CoordinationTrainer:
    def __init__(self, cost, world_model, lr=3e-3, iters=3000, batch_size=512, seed=0):
        self.cost, self.wm = cost, world_model
        self.iters, self.batch_size = iters, batch_size
        self.opt = torch.optim.Adam(cost.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    @torch.no_grad()
    def _dataset(self, segments):
        B, H = segments.actions.shape
        imagined = self.wm.rollout(self.wm.encode(segments.obs), segments.actions)
        real = self.wm.target_encoder(segments.visited.flatten(0, 1)).view(B, H, -1)
        m = segments.alive
        x = torch.cat([imagined[m], real[m]])
        y = torch.cat([segments.events[m], segments.events[m]])
        return x, y

    def fit(self, segments, verbose=True, eval_every=500):
        x, y = self._dataset(segments)
        for p in self.cost.parameters():
            p.requires_grad = True
        self.cost.train()
        for it in range(1, self.iters + 1):
            i = torch.randint(len(x), (self.batch_size,), generator=self.gen)
            loss = F.binary_cross_entropy_with_logits(self.cost.body(x[i]), y[i])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if it % eval_every == 0 or it == 1:
                self.log.append({"iter": it, "loss": loss.item()})
                if verbose:
                    print(f"  [coordination] {it:5d}/{self.iters}  perte {loss.item():.4f}",
                          flush=True)
        self.cost.eval()
        return self.log
