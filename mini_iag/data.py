"""Collecte de données dans le monde en grille (agent qui marche au hasard).

Graines : la carte m de la collecte de graine `seed` utilise la graine
seed * 10 000 + m. Graine 0 = cartes d'entraînement, graine 1 = cartes de
test JAMAIS VUES pendant l'entraînement.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .environment import GridWorld

TRAIN_SEED, TEST_SEED = 0, 1


@dataclass
class Transitions:
    obs: torch.Tensor        # (B, 4, n, n)
    action: torch.Tensor     # (B,)
    next_obs: torch.Tensor   # (B, 4, n, n)

    def __len__(self):
        return len(self.action)

    def batch(self, size, generator=None):
        """(obs, action, obs suivante, événements perçus dans l'obs suivante)."""
        i = torch.randint(len(self), (size,), generator=generator)
        return (self.obs[i].float(), self.action[i], self.next_obs[i].float(),
                self.next_events[i])

    @property
    def next_events(self):
        """(B, 2) : l'agent est-il sur la lave / sur l'objectif après l'action ?"""
        if not hasattr(self, "_events"):
            from .metrics import labels
            self._events = torch.stack(labels(self.next_obs), 1)
        return self._events


@dataclass
class Segments:
    """Morceaux de trajectoires de longueur H, pour apprendre à imaginer plusieurs pas.
    Après la fin d'un épisode, les pas restants sont masqués (alive = False)."""
    obs: torch.Tensor        # (B, 4, n, n)       état de départ
    actions: torch.Tensor    # (B, H)
    visited: torch.Tensor    # (B, H, 4, n, n)    états atteints
    events: torch.Tensor     # (B, H, 2)          lave / objectif perçus
    alive: torch.Tensor      # (B, H)             pas valide ?

    def __len__(self):
        return len(self.actions)

    def batch(self, size, generator=None):
        i = torch.randint(len(self), (size,), generator=generator)
        return self.obs[i], self.actions[i], self.visited[i], self.events[i], self.alive[i]


def collect(cfg, n_maps=300, steps=30, seed=TRAIN_SEED):
    """Transitions (obs, action, obs suivante) d'un agent aléatoire."""
    rng = np.random.default_rng(seed)
    env = GridWorld(cfg)
    obs, act, nxt = [], [], []
    for m in range(n_maps):
        o = env.reset(seed=seed * 10_000 + m)
        for _ in range(steps):
            a = int(rng.integers(cfg.n_actions))
            o2, _, done, _ = env.step(a)
            obs.append(o); act.append(a); nxt.append(o2)
            o = o2
            if done:
                o = env.reset(seed=seed * 10_000 + m)
    return Transitions(torch.tensor(np.array(obs)), torch.tensor(act),
                       torch.tensor(np.array(nxt)))


def collect_sequences(cfg, n_seq=1000, horizon=5, seed=TEST_SEED):
    """Séquences de `horizon` actions sans fin d'épisode, pour tester l'imagination.
    Renvoie (obs de départ (B,4,n,n), actions (B,H), obs visitées (B,H,4,n,n))."""
    rng = np.random.default_rng(seed + 500)
    env = GridWorld(cfg)
    starts, actions, visited = [], [], []
    m = 0
    while len(starts) < n_seq:
        o0 = env.reset(seed=seed * 10_000 + m)
        m += 1
        acts = rng.integers(cfg.n_actions, size=horizon)
        seq, ok = [], True
        for a in acts:
            o, _, done, _ = env.step(int(a))
            seq.append(o)
            if done:
                ok = False
                break
        if ok:
            starts.append(o0); actions.append(acts); visited.append(seq)
    return (torch.tensor(np.array(starts)), torch.tensor(np.array(actions)),
            torch.tensor(np.array(visited)))


def collect_event_sequences(cfg, n_seq=3000, horizon=5, seed=TEST_SEED):
    """Séquences d'actions aléatoires qui PEUVENT finir sur la lave ou l'objectif.
    Pour tester l'anticipation : le modèle prévoit-il l'événement h pas à l'avance ?
    Renvoie (obs de départ, actions (B,H), événements (B,H,2), vivant (B,H)) ;
    vivant[:, h] = l'épisode n'était pas fini avant le pas h."""
    from .metrics import labels
    rng = np.random.default_rng(seed + 700)
    env = GridWorld(cfg)
    starts, actions, events, alive = [], [], [], []
    for m in range(n_seq):
        o0 = env.reset(seed=seed * 10_000 + m)
        acts = rng.integers(cfg.n_actions, size=horizon)
        ev, al, done = np.zeros((horizon, 2), np.float32), np.zeros(horizon, bool), False
        for h, a in enumerate(acts):
            al[h] = not done
            if done:
                continue
            o, _, done, _ = env.step(int(a))
            lava, goal = labels(torch.tensor(o[None]))
            ev[h] = [lava.item(), goal.item()]
        starts.append(o0); actions.append(acts); events.append(ev); alive.append(al)
    return (torch.tensor(np.array(starts)), torch.tensor(np.array(actions)),
            torch.tensor(np.array(events)), torch.tensor(np.array(alive)))


def collect_segments(cfg, n_maps=2000, per_map=10, horizon=3, seed=TRAIN_SEED):
    """Segments de trajectoires aléatoires (peuvent finir sur lave/objectif)."""
    from .metrics import labels
    rng = np.random.default_rng(seed + 300)
    env = GridWorld(cfg)
    obs0, acts, vis, evs, alv = [], [], [], [], []
    for m in range(n_maps):
        for _ in range(per_map):
            o0 = env.reset(seed=seed * 10_000 + m)
            for _ in range(int(rng.integers(0, 6))):     # départ un peu plus loin dans la carte
                o0, _, done, _ = env.step(int(rng.integers(cfg.n_actions)))
                if done:
                    o0 = env.reset(seed=seed * 10_000 + m)
            a = rng.integers(cfg.n_actions, size=horizon)
            v = np.repeat(o0[None], horizon, 0)
            ev, al, done, o = np.zeros((horizon, 2), np.float32), np.zeros(horizon, bool), False, o0
            for h in range(horizon):
                if not done:
                    al[h] = True
                    o, _, done, _ = env.step(int(a[h]))
                    lava, goal = labels(torch.tensor(o[None]))
                    ev[h] = [lava.item(), goal.item()]
                v[h] = o
            obs0.append(o0); acts.append(a); vis.append(v); evs.append(ev); alv.append(al)
    return Segments(torch.tensor(np.array(obs0)), torch.tensor(np.array(acts)),
                    torch.tensor(np.array(vis)), torch.tensor(np.array(evs)),
                    torch.tensor(np.array(alv)))
