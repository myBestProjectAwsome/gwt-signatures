"""La critique d'actions apprend en vivant (Q-learning sur l'expérience vécue).

Jusqu'ici, la critique n'apprenait que sur l'expérience d'origine (un agent qui
marchait au hasard), où certaines situations sont très rares : porter la clé
longtemps, ouvrir une porte. En vivant, elle ne progressait plus.

Ici, chaque séance d'apprentissage continue aussi son Q-learning, sur le même
mélange que l'imagination : moitié souvenirs récents, moitié souvenirs anciens.

Chaque transition vécue sert à TOUS les événements à la fois (objectif, clé,
porte) : c'est l'idée du « rejeu a posteriori » (Hindsight Experience Replay).
Si l'agent ouvre une porte par hasard en cherchant l'objectif, la critique
apprend quand même comment on ouvre une porte. Aucune tâche ne lui demande
jamais d'en ouvrir une : elle l'apprend de ce qui lui arrive.

Comme au départ : états RÉELS seulement, jamais imaginés, et une copie lente
(moyenne mobile) de la critique sert de cible.
"""
import copy

import torch
from torch.nn import functional as F

from ..data_keydoor import EVENT_ORDER
from ..modules.action_critic import TARGETS

REWARD = [EVENT_ORDER.index(t) for t in TARGETS]
LAVA = EVENT_ORDER.index("lava")


class CriticLearner:
    def __init__(self, critic, world_model, buffer, lr=1e-4, batch_size=256, steps_per_update=40,
                 discount=0.9, ema=0.99, old_fraction=0.5):
        self.critic, self.wm, self.buffer = critic, world_model, buffer
        self.target = copy.deepcopy(critic)
        for p in self.target.parameters():
            p.requires_grad = False
        for p in critic.parameters():
            p.requires_grad = True
        self.opt = torch.optim.Adam(critic.parameters(), lr=lr)
        self.batch_size, self.steps, self.discount = batch_size, steps_per_update, discount
        self.ema, self.old_fraction = ema, old_fraction

    def transitions(self):
        """Toutes les transitions valides d'un lot de segments : (z, action, z suivant, événements)."""
        obs, actions, visited, events, alive = self.buffer.batch(self.batch_size, self.old_fraction)
        B, H = actions.shape
        with torch.no_grad():
            z_vis = self.wm.encode(visited.flatten(0, 1)).view(B, H, -1)
            z_prev = torch.cat([self.wm.encode(obs).unsqueeze(1), z_vis[:, :-1]], 1)
        m = alive.flatten()
        return (z_prev.flatten(0, 1)[m], actions.flatten()[m], z_vis.flatten(0, 1)[m],
                events.flatten(0, 1)[m])

    def loss(self, z, a, z_next, ev):
        r = ev[:, REWARD]
        with torch.no_grad():
            v_next = self.target(z_next).max(-1).values
            y = r + self.discount * (1 - r) * (1 - ev[:, LAVA]).unsqueeze(1) * v_next
        pred = self.critic(z)[torch.arange(len(a)), :, a]
        return F.mse_loss(pred, y)

    def update(self):
        self.critic.train()
        total = 0.0
        for _ in range(self.steps):
            loss = self.loss(*self.transitions())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for pt, p in zip(self.target.parameters(), self.critic.parameters()):
                    pt.mul_(self.ema).add_(p, alpha=1 - self.ema)
            total += loss.item()
        self.critic.eval()
        return total / self.steps

    def state(self):
        return {"critic": self.critic.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.opt.state_dict()}

    def load_state(self, state):
        self.critic.load_state_dict(state["critic"])
        self.target.load_state_dict(state["target"])
        self.opt.load_state_dict(state["optimizer"])
