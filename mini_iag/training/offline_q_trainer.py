"""Q-learning hors ligne de la critique d'actions, sur l'expérience réelle.

Données : les transitions vécues (obs, action, obs suivante, événements perçus).
Pour chaque événement visé e, équation de Bellman sur des états RÉELS :
    Q_e(z, a) <- R_e + discount x (1 - R_e) x (1 - lave) x max_a' Q_e(z', a')
où R_e = 1 si l'événement e a eu lieu pendant la transition. Le « max » fait
apprendre le mieux qu'on aurait pu faire, même à partir d'une expérience où
l'agent marchait au hasard. Une copie lente (moyenne mobile) sert de cible.
Aucune imagination : pas d'erreurs qui s'accumulent.
"""
import copy

import torch
from torch.nn import functional as F

from ..data_keydoor import EVENT_ORDER
from ..modules.action_critic import TARGETS


class OfflineQTrainer:
    def __init__(self, critic, world_model, discount=0.9, lr=3e-4, iters=30000,
                 batch_size=512, ema=0.995, seed=0):
        self.critic, self.wm = critic, world_model
        self.target = copy.deepcopy(critic)
        for p in self.target.parameters():
            p.requires_grad = False
        self.discount, self.iters, self.batch_size, self.ema = discount, iters, batch_size, ema
        self.opt = torch.optim.Adam(critic.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    @torch.no_grad()
    def _encode(self, obs, chunk=20000):
        return torch.cat([self.wm.encode(obs[i:i + chunk].float()) for i in range(0, len(obs), chunk)])

    def fit(self, transitions, verbose=True, eval_every=5000):
        z, z_next = self._encode(transitions.obs), self._encode(transitions.next_obs)
        ev = transitions.next_events.float()
        reward = ev[:, [EVENT_ORDER.index(t) for t in TARGETS]]
        alive = 1 - ev[:, EVENT_ORDER.index("lava")]
        act = transitions.action
        rows = torch.arange(self.batch_size)
        self.critic.train()
        for it in range(1, self.iters + 1):
            i = torch.randint(len(z), (self.batch_size,), generator=self.gen)
            with torch.no_grad():
                v_next = self.target(z_next[i]).max(-1).values
                y = reward[i] + self.discount * (1 - reward[i]) * alive[i].unsqueeze(1) * v_next
            pred = self.critic(z[i])[rows, :, act[i]]
            loss = F.mse_loss(pred, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for pt, p in zip(self.target.parameters(), self.critic.parameters()):
                    pt.mul_(self.ema).add_(p, alpha=1 - self.ema)
            if it % eval_every == 0 or it == 1:
                self.log.append({"iter": it, "loss": loss.item()})
                if verbose:
                    print(f"  [critique d'actions] {it:5d}/{self.iters}  perte {loss.item():.5f}",
                          flush=True)
        self.critic.eval()
        return self.log
