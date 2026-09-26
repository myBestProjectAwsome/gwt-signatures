"""Collecte de données dans le monde v2 (clé, porte), cartes standard publiques.

Un agent marche au hasard pendant `steps` pas (l'épisode ne s'arrête que sur la
lave). Chaque transition est étiquetée par les événements PERÇUS : objectif,
clé, porte, lave. Ce sont des faits de la physique du monde, pas des tâches :
aucune récompense n'est attachée ici.

Graines : mêmes conventions que data.py (graine 0 = entraînement, 1 = contrôle).
"""
import numpy as np
import torch

from .data import TRAIN_SEED, Segments, Transitions
from .environment.keydoor_world import EVENTS, KeyDoorWorld

EVENT_ORDER = ("goal", "key", "door", "lava")      # = ordre des sorties du coût et du modèle
assert set(EVENT_ORDER) == set(EVENTS)


def encode_events(events):
    return [float(e in events) for e in EVENT_ORDER]


def collect_kd(cfg, n_maps=3000, steps=50, seed=TRAIN_SEED):
    rng = np.random.default_rng(seed + 11)
    world = KeyDoorWorld(cfg)
    obs, act, nxt, ev = [], [], [], []
    for m in range(n_maps):
        o = world.reset(seed=seed * 10_000 + m)
        for _ in range(steps):
            a = int(rng.integers(cfg.n_actions))
            o2, events, dead = world.step(a)
            obs.append(o); act.append(a); nxt.append(o2); ev.append(encode_events(events))
            o = o2
            if dead:
                o = world.reset(seed=seed * 10_000 + m)
    # stockage compact (0/1 sur 8 bits) : 4x moins de mémoire ; converti en float par lot
    t = Transitions(torch.tensor(np.array(obs, dtype=np.uint8)), torch.tensor(act),
                    torch.tensor(np.array(nxt, dtype=np.uint8)))
    t._events = torch.tensor(ev)
    return t


def collect_kd_segments(cfg, n_maps=2000, per_map=10, horizon=6, seed=TRAIN_SEED):
    """Segments de trajectoires aléatoires (peuvent finir dans la lave)."""
    rng = np.random.default_rng(seed + 311)
    world = KeyDoorWorld(cfg)
    obs0, acts, vis, evs, alv = [], [], [], [], []
    for m in range(n_maps):
        for _ in range(per_map):
            o0 = world.reset(seed=seed * 10_000 + m)
            for _ in range(int(rng.integers(0, 15))):          # départ plus loin dans la carte
                o0, _, dead = world.step(int(rng.integers(cfg.n_actions)))
                if dead:
                    o0 = world.reset(seed=seed * 10_000 + m)
            a = rng.integers(cfg.n_actions, size=horizon)
            v = np.repeat(o0[None], horizon, 0)
            e = np.zeros((horizon, len(EVENT_ORDER)), np.float32)
            al, dead, o = np.zeros(horizon, bool), False, o0
            for h in range(horizon):
                if not dead:
                    al[h] = True
                    o, events, dead = world.step(int(a[h]))
                    e[h] = encode_events(events)
                v[h] = o
            obs0.append(o0); acts.append(a); vis.append(v); evs.append(e); alv.append(al)
    return Segments(torch.tensor(np.array(obs0)), torch.tensor(np.array(acts)),
                    torch.tensor(np.array(vis)), torch.tensor(np.array(evs)),
                    torch.tensor(np.array(alv)))
