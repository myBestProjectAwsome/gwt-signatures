"""Mesures communes aux diagnostics de la mini-IAG (étapes 1, 2, ...)."""
import numpy as np
import torch
from torch.nn import functional as F

from .environment.gridworld import AGENT, GOAL, LAVA, MOVES, WALL


# ---------------------------------------------------------------- étiquettes
def agent_cell(obs):
    """Indice de la case de l'agent (0..n*n-1) et nombre de cases."""
    n = obs.shape[-1]
    return obs[:, AGENT].reshape(len(obs), -1).argmax(1), n * n


def labels(obs):
    """(sur la lave, sur l'objectif) pour chaque observation, en 0/1."""
    on = lambda ch: (obs[:, ch] * obs[:, AGENT]).sum((1, 2)) > 0
    return on(LAVA).float(), on(GOAL).float()


def equivalent_actions(obs, act):
    """Masque (B, 4) des actions qui mènent au même résultat que l'action réelle
    (deux actions bloquées par un mur laissent toutes deux l'agent sur place)."""
    n = obs.shape[-1]
    cell = obs[:, AGENT].reshape(len(obs), -1).argmax(1)
    r, c = cell // n, cell % n
    dest = []
    for dr, dc in MOVES:
        rr, cc = r + dr, c + dc
        blocked = obs[torch.arange(len(obs)), WALL, rr, cc] > 0
        dest.append(torch.where(blocked, cell, rr * n + cc))
    dest = torch.stack(dest, 1)
    return dest == dest[torch.arange(len(obs)), act].unsqueeze(1)


# ---------------------------------------------------------------- mesures
def auc(scores, y):
    """Aire sous la courbe ROC (0.5 = hasard, 1 = parfait), par les rangs."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = y.bool()
    n1, n0 = pos.sum().item(), (~pos).sum().item()
    return (ranks[pos].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fit_linear_probe(x, y, n_classes, ridge=1e-2):
    """Régression ridge (forme fermée) vers un one-hot. Renvoie les poids."""
    x = torch.cat([x, torch.ones(len(x), 1)], 1)
    Y = F.one_hot(y, n_classes).float()
    A = x.T @ x + ridge * torch.eye(x.shape[1])
    return torch.linalg.solve(A, x.T @ Y)


def apply_linear_probe(W, x):
    return (torch.cat([x, torch.ones(len(x), 1)], 1) @ W).argmax(1)


def linear_probe(x, y, n_classes, split=0.7, ridge=1e-2):
    """Précision test d'une sonde linéaire (70 % entraînement / 30 % test)."""
    k = int(split * len(x))
    W = fit_linear_probe(x[:k], y[:k], n_classes, ridge)
    return (apply_linear_probe(W, x[k:]) == y[k:]).float().mean().item()


@torch.no_grad()
def action_identification(world_model, obs, act, next_obs):
    """S1 : connaissant z_t et z_{t+1}, retrouve-t-on l'action faite ?
    Renvoie (précision, hasard, erreur de prédiction, erreur 'rien ne change')."""
    z, z_next = world_model.encode(obs), world_model.target_encoder(next_obs)
    preds = torch.stack([world_model.predict(z, torch.full_like(act, a))
                         for a in range(4)], 1)
    dist = ((preds - z_next.unsqueeze(1)) ** 2).sum(-1)
    equiv = equivalent_actions(obs, act)
    idx = torch.arange(len(act))
    return (equiv[idx, dist.argmin(1)].float().mean().item(),
            equiv.float().mean().item(),
            ((preds[idx, act] - z_next) ** 2).mean().item(),
            ((z - z_next) ** 2).mean().item())


@torch.no_grad()
def workspace_selection(ws, d, n_tokens=8, batch=2000, seed=0):
    """S4 (étape 1) : un seul contenu porte un signal fixe, les autres sont du bruit.
    Renvoie (argmax juste, masse d'attention sur le bon, entropie normalisée)."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, n_tokens, d, generator=g)
    signal = torch.randn(d, generator=g)
    target = torch.randint(n_tokens, (batch,), generator=g)
    tokens[torch.arange(batch), target] += 2 * signal
    _, _, attn = ws(tokens)
    att = attn.mean(1)
    acc = (att.argmax(1) == target).float().mean().item()
    ent = -(att * att.clamp_min(1e-9).log()).sum(1).mean().item() / np.log(n_tokens)
    return acc, att[torch.arange(batch), target].mean().item(), ent
