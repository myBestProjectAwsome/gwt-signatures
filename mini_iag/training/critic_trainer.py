"""Entraînement de la critique par itération de valeur dans l'imagination.

Pour un état latent z, on imagine les 4 actions possibles avec le modèle du
monde (z' = prédicteur(z, a)), on lit les événements prévus par le module de
coût, et on applique l'équation de Bellman, pour chaque événement visé e :
    Q_e(z, a) = P_e(z') + discount x (1 - P_e(z')) x (1 - P_lave(z')) x V_e(z')
    V_e(z)   <- max_a Q_e(z, a)
Atteindre e rapporte 1 ; la lave arrête tout ; sinon on garde la valeur
future, atténuée. Aucune information sur les vrais chemins n'est utilisée :
tout vient du modèle du monde et du module de coût.

Une copie lente de la critique (moyenne mobile) sert de cible, comme pour
l'encodeur cible du JEPA, pour stabiliser l'apprentissage.
"""
import copy

import torch
from torch.nn import functional as F

from ..modules.configurable_cost import EVENT_INDEX
from ..modules.critic import TARGETS


class CriticTrainer:
    def __init__(self, critic, world_model, cost, discount=0.9, lr=1e-3, iters=6000,
                 batch_size=512, ema=0.99, seed=0):
        self.critic, self.wm, self.cost = critic, world_model, cost
        self.target = copy.deepcopy(critic)
        for p in self.target.parameters():
            p.requires_grad = False
        self.discount, self.iters, self.batch_size, self.ema = discount, iters, batch_size, ema
        self.opt = torch.optim.Adam(critic.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    @torch.no_grad()
    def _bellman(self, z, n_actions=4):
        B = len(z)
        acts = torch.arange(n_actions).repeat(B)
        z_rep = z.repeat_interleave(n_actions, 0)
        z_next = self.wm.predict(z_rep, acts)
        x = torch.cat([z_rep, z_next], dim=-1) if getattr(self.cost, "pair_input", False) else z_next
        p = self.cost.events(x)
        reach = p[:, [EVENT_INDEX[t] for t in TARGETS]]
        alive = 1 - p[:, EVENT_INDEX["lava"]:EVENT_INDEX["lava"] + 1]
        q = reach + self.discount * (1 - reach) * alive * self.target(z_next)
        return q.view(B, n_actions, len(TARGETS)).max(1).values, z_next

    def fit(self, latents, verbose=True, eval_every=1000):
        self.critic.train()
        for it in range(1, self.iters + 1):
            i = torch.randint(len(latents), (self.batch_size,), generator=self.gen)
            z = latents[i]
            target, z_next = self._bellman(z)
            # on apprend aussi sur des états imaginés : c'est là que le planificateur l'interroge
            j = torch.randint(len(z_next), (self.batch_size // 2,), generator=self.gen)
            target_next, _ = self._bellman(z_next[j])
            loss = F.mse_loss(self.critic(torch.cat([z, z_next[j]])),
                              torch.cat([target, target_next]))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for pt, p in zip(self.target.parameters(), self.critic.parameters()):
                    pt.mul_(self.ema).add_(p, alpha=1 - self.ema)
            if it % eval_every == 0 or it == 1:
                self.log.append({"iter": it, "loss": loss.item()})
                if verbose:
                    print(f"  [critique] {it:5d}/{self.iters}  perte {loss.item():.5f}", flush=True)
        self.critic.eval()
        return self.log
