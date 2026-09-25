"""Entraînement du modèle du monde (étape 2a).

Auto-supervisé : seulement l'expérience de l'agent, des transitions
(observation, action, observation suivante, événements perçus) collectées
par un agent qui marche au hasard.
"""
import torch

from ..metrics import action_identification


class WorldModelTrainer:
    def __init__(self, world_model, lr=1e-3, iters=6000, batch_size=256,
                 var_weight=1.0, cov_weight=0.04, inv_weight=1.0, event_weight=1.0,
                 multistep_weight=1.0, seed=0):
        self.wm = world_model
        self.iters, self.batch_size = iters, batch_size
        self.weights = dict(var_weight=var_weight, cov_weight=cov_weight,
                            inv_weight=inv_weight, event_weight=event_weight)
        self.multistep_weight = multistep_weight
        params = [p for p in world_model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, iters)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    def fit(self, train, test=None, segments=None, eval_every=500, verbose=True):
        """train : Transitions (1 pas) ; segments : Segments (H pas), optionnel."""
        self.wm.train()
        for it in range(1, self.iters + 1):
            total, terms = self.wm.loss(*train.batch(self.batch_size, self.gen), **self.weights)
            if segments is not None and self.multistep_weight > 0:
                terms["multistep"] = self.wm.multistep_loss(
                    *segments.batch(self.batch_size, self.gen),
                    event_weight=self.weights["event_weight"])
                total = total + self.multistep_weight * terms["multistep"]
            self.opt.zero_grad()
            total.backward()
            self.opt.step()
            self.sched.step()
            self.wm.update_target()
            if it % eval_every == 0 or it == 1:
                row = {"iter": it, **{k: v.item() for k, v in terms.items()}}
                if test is not None:
                    self.wm.eval()
                    row["S1"] = action_identification(self.wm, test.obs, test.action,
                                                      test.next_obs)[0]
                    self.wm.train()
                self.log.append(row)
                if verbose:
                    s1 = f"  S1 {100 * row['S1']:5.1f} %" if "S1" in row else ""
                    print(f"  [modèle du monde] {it:5d}/{self.iters}  prédiction "
                          f"{row['prediction']:.4f}  inverse {row['inverse']:.3f}{s1}", flush=True)
        self.wm.eval()
        return self.log
