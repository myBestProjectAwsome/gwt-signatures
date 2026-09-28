"""Q-learning hors ligne de la carte mentale, sur l'expérience réelle.

Même équation que la critique d'actions (offline_q_trainer.py), pour chaque
événement visé e :
    Q_e(s, a) <- R_e + discount x (1 - R_e) x (1 - lave) x max_a' Q_e(s', a')
Seule la forme du réseau change : la carte mentale propage les valeurs de case
en case au lieu de les deviner en une passe.

Option cell_weight > 0 : en plus de Bellman, chaque transition vécue renseigne
directement la case où l'agent est arrivé (événement ? mort ?) et dit s'il est
resté bloqué. C'est ce que l'agent CONSTATE en marchant, rien de plus.

Option relative=True (essayée, NON retenue) : une erreur relative, pour mieux
séparer les petites valeurs loin de la cible. L'apprentissage devient instable
(la direction d'une action est mal apprise) et les choix tombent à 30-40 %
d'actions optimales. On garde l'erreur absolue (MSE).
"""
import copy

import torch
from torch.nn import functional as F

from ..data_keydoor import EVENT_ORDER
from ..environment.keydoor_world import AGENT
from ..modules.mental_map import TARGETS


class MapTrainer:
    def __init__(self, mental_map, lr=1e-3, iters=15000, batch_size=256, ema=0.995, seed=0,
                 relative=False, cell_weight=0.0, focal=0.0):
        self.map = mental_map
        self.target = copy.deepcopy(mental_map)
        for p in self.target.parameters():
            p.requires_grad = False
        self.iters, self.batch_size, self.ema = iters, batch_size, ema
        self.relative, self.cell_weight, self.focal = relative, cell_weight, focal
        self.opt = torch.optim.Adam(mental_map.parameters(), lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    def fit(self, transitions, verbose=True, eval_every=2500):
        ev = transitions.next_events.float()
        reward = ev[:, [EVENT_ORDER.index(t) for t in TARGETS]]
        alive = 1 - ev[:, EVENT_ORDER.index("lava")]
        act, gamma = transitions.action, self.map.discount
        rows = torch.arange(self.batch_size)
        self.map.train()
        for it in range(1, self.iters + 1):
            i = torch.randint(len(act), (self.batch_size,), generator=self.gen)
            with torch.no_grad():
                v_next = self.target(transitions.next_obs[i].float()).max(-1).values
                y = reward[i] + gamma * (1 - reward[i]) * alive[i].unsqueeze(1) * v_next
            pred = self.map(transitions.obs[i].float())[rows, :, act[i]]
            if self.relative:
                # erreur RELATIVE : à 15 pas, deux actions ne diffèrent que de quelques %
                # d'une valeur déjà petite ; une erreur absolue les confondrait
                loss = (((pred - y) / (y + 0.02)) ** 2).mean()
            else:
                loss = F.mse_loss(pred, y)
            if self.cell_weight > 0:
                loss = loss + self.cell_weight * self.cell_loss(transitions, i, ev, act)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for pt, p in zip(self.target.parameters(), self.map.parameters()):
                    pt.mul_(self.ema).add_(p, alpha=1 - self.ema)
            if it % eval_every == 0 or it == 1:
                self.log.append({"iter": it, "loss": loss.item()})
                if verbose:
                    print(f"  [carte mentale] {it:5d}/{self.iters}  perte {loss.item():.5f}", flush=True)
        self.map.eval()
        return self.log

    def cell_loss(self, transitions, i, ev, act):
        """Ce que l'agent a CONSTATÉ en marchant : la case où il est arrivé a-t-elle
        déclenché un événement ? l'a-t-elle tué ? est-il resté sur place ?
        Chaque transition vécue renseigne la case concernée (auto-supervision locale)."""
        obs, nxt = transitions.obs[i].float(), transitions.next_obs[i].float()
        r, b, d = self.map.maps(obs)
        here, there = obs[:, AGENT], nxt[:, AGENT]
        moved = (here != there).flatten(1).any(1).float()
        e = ev[i]
        at = lambda m: (m * there[:, None]).sum((-1, -2))                     # valeur à la case d'arrivée
        reward = e[:, [EVENT_ORDER.index(t) for t in TARGETS]]
        lava = e[:, EVENT_ORDER.index("lava")]
        def bce(p, y, reduction="mean"):
            """Entropie croisée, éventuellement « focale » : les cas que la carte se trompe
            (souvent les cas rares, comme une porte fermée) pèsent plus (Lin et al., 2017)."""
            loss = F.binary_cross_entropy(p, y, reduction="none")
            if self.focal > 0:
                pt = torch.where(y > 0.5, p, 1 - p)
                loss = loss * (1 - pt) ** self.focal
            return loss if reduction == "none" else loss.mean()
        w = moved[:, None]
        loss_r = (bce(at(r).clamp(1e-5, 1 - 1e-5), reward, reduction="none") * w).sum() / w.sum().clamp_min(1) / 3
        loss_d = (bce(at(d)[:, 0].clamp(1e-5, 1 - 1e-5), lava, reduction="none") * moved).sum() / moved.sum().clamp_min(1)
        stuck = (self.map._shift(b, self.map.kernels()) * here[:, None, None]).sum((-1, -2))[:, 0]   # (B, A)
        stuck = stuck[torch.arange(len(act[i])), act[i]].clamp(1e-5, 1 - 1e-5)
        loss_b = bce(stuck, 1 - moved)
        return loss_r + loss_d + loss_b
