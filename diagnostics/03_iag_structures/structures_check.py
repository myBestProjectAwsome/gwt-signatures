"""
structures_check.py — Mini-IAG, étape 1 : l'architecture vide.

Question : à l'initialisation aléatoire, que sait faire chaque module ?
La feuille de route dit "le système ne produit que du bruit". On le mesure,
module par module, pour avoir des chiffres de référence (le "avant") à
comparer après l'entraînement de l'étape 2.

Données : transitions (obs, action, obs suivante) collectées par un agent
qui marche au hasard sur 300 cartes aléatoires.

Mesures :
  S1  modèle du monde : connaissant z_t et z_{t+1}, peut-il dire QUELLE action
      a été faite ? On prédit l'état suivant pour chacune des 4 actions et on
      garde la plus proche. (Deux actions bloquées par un mur donnent le même
      résultat : elles comptent toutes deux comme justes.)
  S2  information dans le latent : peut-on retrouver la case de l'agent à partir
      du vecteur latent par une simple régression linéaire (sonde) ?
  S3  module de coût : AUC pour détecter la lave et l'objectif (0.5 = hasard)
  S4  espace de travail : parmi 8 contenus dont un seul est pertinent,
      l'attention se porte-t-elle sur le bon ? (hasard = 12.5 %)
  S5  mémoire : retrouve-t-elle un état déjà stocké ?
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mini_iag import Architecture, Config, GridWorld  # noqa: E402
from mini_iag.environment.gridworld import AGENT, GOAL, LAVA  # noqa: E402


def collect(cfg, n_maps=300, steps=30, seed=0):
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
    return (torch.tensor(np.array(obs)), torch.tensor(act), torch.tensor(np.array(nxt)))


def agent_cell(obs):
    n = obs.shape[-1]
    return obs[:, AGENT].reshape(len(obs), -1).argmax(1), n * n


def equivalent_actions(obs, act):
    """Masque (B, 4) des actions qui mènent au même résultat que l'action réelle."""
    from mini_iag.environment.gridworld import MOVES, WALL
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


def labels(obs):
    on = lambda ch: (obs[:, ch] * obs[:, AGENT]).sum((1, 2)) > 0
    return on(LAVA).float(), on(GOAL).float()


def auc(scores, y):
    """Aire sous la courbe ROC (0.5 = hasard, 1 = parfait), par les rangs."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = y.bool()
    n1, n0 = pos.sum().item(), (~pos).sum().item()
    return (ranks[pos].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def linear_probe(x, y, n_classes, split=0.7, ridge=1e-2):
    """Régression ridge (forme fermée) vers un one-hot ; renvoie la précision test."""
    k = int(split * len(x))
    x = torch.cat([x, torch.ones(len(x), 1)], 1)
    Y = torch.nn.functional.one_hot(y, n_classes).float()
    A = x[:k].T @ x[:k] + ridge * torch.eye(x.shape[1])
    W = torch.linalg.solve(A, x[:k].T @ Y[:k])
    return ((x[k:] @ W).argmax(1) == y[k:]).float().mean().item()


def workspace_selection(ws, d, n_tokens=8, batch=2000, seed=0):
    """Un seul contenu porte un signal fixe ; les autres sont du bruit."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(batch, n_tokens, d, generator=g)
    signal = torch.randn(d, generator=g)
    target = torch.randint(n_tokens, (batch,), generator=g)
    tokens[torch.arange(batch), target] += 2 * signal
    _, _, attn = ws(tokens)                       # (B, K, N)
    att = attn.mean(1)                            # moyenne sur les slots
    acc = (att.argmax(1) == target).float().mean().item()
    ent = -(att * att.clamp_min(1e-9).log()).sum(1).mean().item() / np.log(n_tokens)
    return acc, att[torch.arange(batch), target].mean().item(), ent


if __name__ == "__main__":
    cfg = Config()
    arch = Architecture(cfg)
    arch.eval()
    obs, act, nxt = collect(cfg)
    lava, goal = labels(nxt)

    with torch.no_grad():
        wm = arch.world_model
        z, z_next = wm.encode(obs), wm.target_encoder(nxt)
        preds = torch.stack([wm.predict(z, torch.full_like(act, a))
                             for a in range(cfg.n_actions)], 1)          # (B, 4, D)
        dist = ((preds - z_next.unsqueeze(1)) ** 2).sum(-1)
        equiv = equivalent_actions(obs, act)
        act_acc = equiv[torch.arange(len(act)), dist.argmin(1)].float().mean().item()
        act_chance = equiv.float().mean().item()
        err_pred = ((preds[torch.arange(len(act)), act] - z_next) ** 2).mean().item()
        err_copy = ((z - z_next) ** 2).mean().item()
        z_all = wm.encode(nxt)
        cell, n_cells = agent_cell(nxt)
        probe_latent = linear_probe(z_all, cell, n_cells)
        probe_raw = linear_probe(nxt.reshape(len(nxt), -1), cell, n_cells)
        chance = torch.bincount(cell, minlength=n_cells).max().item() / len(cell)
        danger, success, _ = arch.cost(z_all)
        auc_lava, auc_goal = auc(danger, lava), auc(success, goal)
        ws_acc, ws_att, ws_ent = workspace_selection(arch.workspace, cfg.latent_dim)
        arch.memory.write(z_all[:1000])
        _, idx = arch.memory.recall(z_all[:1000], k=1)
        # un même état peut apparaître plusieurs fois : on compare les contenus
        mem_ok = (z_all[idx[:, 0]] - z_all[:1000]).abs().max(1).values.lt(1e-5).float().mean().item()

    print(f"{len(obs)} transitions, {300} cartes aléatoires, poids aléatoires (graine {cfg.seed})\n")
    print("Paramètres :")
    for k, v in arch.parameter_counts().items():
        print(f"  {k:34} {v:>8,}")
    print("\nS1 modèle du monde")
    print(f"   action retrouvée            {100 * act_acc:.1f} %   (hasard {100 * act_chance:.1f} %)")
    print(f"   erreur de prédiction        {err_pred:.2e}")
    print(f"   erreur 'rien ne change'     {err_copy:.2e}   (le latent aléatoire bouge à peine)")
    print("S2 case de l'agent retrouvée par une sonde linéaire")
    print(f"   depuis le latent aléatoire  {100 * probe_latent:.1f} %")
    print(f"   depuis l'image brute        {100 * probe_raw:.1f} %")
    print(f"   hasard (classe majoritaire) {100 * chance:.1f} %")
    print("S3 module de coût (AUC, 0.5 = hasard)")
    print(f"   détection de la lave        {auc_lava:.2f}   ({int(lava.sum())} états sur lave)")
    print(f"   détection de l'objectif     {auc_goal:.2f}   ({int(goal.sum())} états sur l'objectif)")
    print("S4 espace de travail (1 contenu pertinent parmi 8)")
    print(f"   attention sur le bon        {ws_att:.3f}   (uniforme 0.125)")
    print(f"   entropie normalisée         {ws_ent:.3f}   (1 = attention uniforme)")
    print(f"   argmax sur le bon           {100 * ws_acc:.1f} %   (trompeur : écarts minuscules, voir README)")
    print("S5 mémoire")
    print(f"   état stocké retrouvé        {100 * mem_ok:.1f} %")
