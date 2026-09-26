"""Tampon d'expérience : ce que l'agent a vécu, plus des souvenirs anciens.

Deux réserves :
  - récente : les épisodes vécus (trajectoires complètes), les plus anciens
    étant oubliés au-delà de la capacité ;
  - ancienne : un échantillon fixe de l'expérience d'origine (l'agent qui
    marchait au hasard, étape 4a).

Chaque lot d'apprentissage mélange les deux (moitié-moitié par défaut).
C'est la parade classique à l'OUBLI CATASTROPHIQUE : un réseau entraîné
seulement sur ses expériences récentes efface ce qu'il savait avant.

Les lots sont des segments de H pas (H = 1 à horizon), au même format que
data.Segments : (obs de départ, actions, états visités, événements, vivant).
"""
import numpy as np
import torch


class ExperienceBuffer:
    def __init__(self, anchor_segments, horizon=5, capacity_steps=60_000, seed=0):
        self.anchor = anchor_segments          # data.Segments (souvenirs anciens)
        self.horizon = horizon
        self.capacity = capacity_steps
        self.episodes = []                     # liste de dicts numpy (trajectoires)
        self.n_steps = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_steps

    def add(self, traj):
        if len(traj["action"]) == 0:
            return
        self.episodes.append(traj)
        self.n_steps += len(traj["action"])
        while self.n_steps > self.capacity and len(self.episodes) > 1:
            self.n_steps -= len(self.episodes.pop(0)["action"])

    def _recent(self, n):
        H = self.horizon
        obs0, acts, vis, evs, alv = [], [], [], [], []
        lengths = np.array([len(e["action"]) for e in self.episodes])
        probs = lengths / lengths.sum()
        for k in self.rng.choice(len(self.episodes), n, p=probs):
            ep = self.episodes[k]
            T = len(ep["action"])
            s = int(self.rng.integers(T))
            idx = np.arange(s, min(s + H, T))
            pad = H - len(idx)
            a = np.concatenate([ep["action"][idx], np.zeros(pad, int)])
            v = np.concatenate([ep["next_obs"][idx], np.repeat(ep["next_obs"][idx[-1:]], pad, 0)])
            e = np.concatenate([ep["events"][idx], np.zeros((pad, ep["events"].shape[1]))])
            al = np.concatenate([np.ones(len(idx), bool), np.zeros(pad, bool)])
            obs0.append(ep["obs"][s]); acts.append(a); vis.append(v); evs.append(e); alv.append(al)
        return (torch.tensor(np.array(obs0)).float(), torch.tensor(np.array(acts)),
                torch.tensor(np.array(vis)).float(), torch.tensor(np.array(evs)).float(),
                torch.tensor(np.array(alv)))

    def _old(self, n):
        i = torch.as_tensor(self.rng.integers(len(self.anchor), size=n))
        s = self.anchor
        return (s.obs[i].float(), s.actions[i], s.visited[i].float(), s.events[i].float(),
                s.alive[i])

    def batch(self, size, old_fraction=0.5):
        n_old = int(size * old_fraction) if self.episodes else size
        parts = []
        if n_old:
            parts.append(self._old(n_old))
        if size - n_old:
            parts.append(self._recent(size - n_old))
        return tuple(torch.cat([p[j] for p in parts]) for j in range(5))

    def state(self):
        return {"episodes": self.episodes, "n_steps": self.n_steps}

    def load_state(self, state):
        self.episodes, self.n_steps = state["episodes"], state["n_steps"]
