"""
planning_check.py — Mini-IAG, étape 3 : l'agent câblé planifie-t-il ?

200 cartes JAMAIS VUES, 30 pas maximum par épisode. Issues possibles :
succès (objectif atteint), lave, bloqué (30 pas écoulés).

Références :
  aléatoire          choisit une action au hasard
  oracle             plus court chemin (BFS) : la borne haute
Agent et ablations (un seul élément change à chaque fois) :
  complet            modèle du monde + coût coordonné + mémoire + workspace
  sans mémoire       ne sait pas s'il est déjà passé quelque part
  sans coordination  coût de l'étape 2 (jamais entraîné sur des états imaginés)
  sans workspace     exécute directement le plan classé premier par le coût
  horizon 1          n'imagine qu'un pas à l'avance
  modèles aléatoires même câblage, modèle du monde NON entraîné (coût coordonné
                     sur ses latents)

Mesures : taux de succès / lave / bloqué ; efficacité = plus court chemin /
pas utilisés (sur les succès) ; fidélité du workspace au classement du coût ;
succès selon la distance de l'objectif.
"""
import copy
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mini_iag import Architecture, Config, GridWorld  # noqa: E402
from mini_iag.agent import Agent  # noqa: E402
from mini_iag.coordinate import DEMO_OFFSET, STEP2, STEP3, coordinate, novelty_scale  # noqa: E402
from mini_iag.data import TEST_SEED, TRAIN_SEED, collect_segments  # noqa: E402
from mini_iag.environment.gridworld import LAVA, MOVES, WALL  # noqa: E402
from mini_iag.train import TRAIN_MAPS, load  # noqa: E402
from mini_iag.training import CoordinationTrainer  # noqa: E402

N_MAPS = 200
OUT = Path(__file__).with_name("planning_results")


# ---------------------------------------------------------------- références
class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, env):
        return int(self.rng.integers(4))


class OracleAgent:
    """Plus court chemin (BFS) qui évite murs et lave."""

    def reset(self):
        pass

    def act(self, env):
        start, parent = env.agent, {env.agent: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == env.goal:
                break
            for a, (dr, dc) in enumerate(MOVES):
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in parent and not env.grid[WALL][nxt] and not env.grid[LAVA][nxt]:
                    parent[nxt] = (cur, a)
                    q.append(nxt)
        node, action = env.goal, 0
        while parent[node] is not None:
            node, action = parent[node]
        return action


def episode(agent, env, map_seed, max_steps, is_model):
    obs = env.reset(seed=map_seed)
    optimal = env.distance_to_goal()
    agent.reset()
    agree = []
    for t in range(1, max_steps + 1):
        if is_model:
            dec = agent.act(obs)
            agree.append(dec.agrees)
            action = dec.action
        else:
            action = agent.act(env)
        obs, _, done, info = env.step(action)
        if done:
            break
    outcome = "succès" if info["success"] else ("lave" if info["danger"] else "bloqué")
    return {"outcome": outcome, "steps": t, "optimal": optimal,
            "agreement": float(np.mean(agree)) if agree else None}


def evaluate(agent, cfg, is_model=True):
    env = GridWorld(cfg)
    return [episode(agent, env, TEST_SEED * 10_000 + DEMO_OFFSET + m, cfg.max_steps, is_model)
            for m in range(N_MAPS)]


def summary(res):
    n = len(res)
    succ = [r for r in res if r["outcome"] == "succès"]
    agree = [r["agreement"] for r in res if r["agreement"] is not None]
    return {k: 100 * sum(r["outcome"] == k for r in res) / n for k in ("succès", "lave", "bloqué")} | {
        "efficacité": float(np.mean([r["optimal"] / r["steps"] for r in succ])) if succ else 0.0,
        "fidélité": 100 * float(np.mean(agree)) if agree else None,
        "par_distance": {d: 100 * float(np.mean([r["outcome"] == "succès" for r in res
                                                  if bucket(r["optimal"]) == d]))
                         for d in BUCKETS},
    }


BUCKETS = ["1", "2", "3", "4", "5", "6+"]


def bucket(d):
    return str(d) if d <= 5 else "6+"


def figure(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    names = list(S)[::-1]
    colors = {"succès": "#2a78d6", "lave": "#eb6834", "bloqué": "#b4b3aa"}
    a = ax[0]
    left = np.zeros(len(names))
    for k in ("succès", "lave", "bloqué"):
        vals = np.array([S[n][k] for n in names])
        a.barh(names, vals, left=left, color=colors[k], edgecolor="white", linewidth=2, label=k)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 7:
                a.text(l0 + v / 2, i, f"{v:.0f} %", ha="center", va="center",
                       color="white" if k != "bloqué" else INK, fontsize=9)
        left += vals
    a.set(title="Issue des épisodes (200 cartes jamais vues)", xlim=(0, 100), xlabel="%")
    a.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    a.spines[["top", "right"]].set_visible(False)

    a = ax[1]
    line_colors = {"oracle": "#8a8980", "complet": "#2a78d6", "sans mémoire": "#1baf7a",
                   "horizon 1": "#eda100", "aléatoire": "#eb6834"}
    x = np.arange(len(BUCKETS))
    ends = []
    for n, col in line_colors.items():
        y = [S[n]["par_distance"][b] for b in BUCKETS]
        a.plot(x, y, marker="o", color=col, lw=2, ms=8, markeredgecolor="white",
               markeredgewidth=2, ls="--" if n == "oracle" else "-", label=n)
        ends.append((y[-1], n, col))
    ends.sort()
    last = -100
    for y, n, col in ends:
        y = max(y, last + 6)
        a.text(len(BUCKETS) - 1 + 0.12, y, n, color=INK, va="center")
        last = y
    a.set(title="Succès selon la distance de l'objectif", xticks=x, xticklabels=BUCKETS,
          xlabel="plus court chemin (pas)", ylabel="succès (%)", ylim=(-3, 105),
          xlim=(-0.2, len(BUCKETS) + 0.6))
    a.grid(axis="y", color=GRID)
    a.spines[["top", "right"]].set_visible(False)
    a.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        figure(json.loads(OUT.with_suffix(".json").read_text()), OUT.with_suffix(".png"))
        sys.exit()
    if not STEP3.exists():
        print("Pas de checkpoint de l'étape 3 : câblage (python -m mini_iag.coordinate)...")
        coordinate(verbose=False)
    ck3 = torch.load(STEP3, weights_only=False)
    cfg = Config(**ck3["config"])
    cue, sigma = ck3["cue"], ck3["novelty_sigma"]
    arch3, _, _ = load(STEP3)
    arch2, _, _ = load(STEP2)

    def agent(arch, **kw):
        return Agent(copy.deepcopy(arch), cfg, cue, sigma, **kw)

    print("Modèles aléatoires : coordination du coût sur un modèle du monde non entraîné...")
    torch.manual_seed(0)
    rand = Architecture(cfg)
    rand.eval()
    CoordinationTrainer(rand.cost, rand.world_model).fit(
        collect_segments(cfg, n_maps=TRAIN_MAPS, horizon=cfg.planning_horizon, seed=TRAIN_SEED),
        verbose=False)
    h1 = copy.deepcopy(cfg)
    h1.planning_horizon = 1

    agents = {
        "oracle": (OracleAgent(), False),
        "complet": (agent(arch3), True),
        "sans workspace": (agent(arch3, use_workspace=False), True),
        "sans mémoire": (agent(arch3, use_memory=False), True),
        "sans coordination": (agent(arch2), True),
        "horizon 1": (Agent(copy.deepcopy(arch3), h1, cue, sigma), True),
        "modèles aléatoires": (Agent(rand, cfg, cue, novelty_scale(rand.world_model, cfg)), True),
        "aléatoire": (RandomAgent(), False),
    }
    S = {}
    for name, (ag, is_model) in agents.items():
        print(f"  évaluation : {name}...", flush=True)
        S[name] = summary(evaluate(ag, cfg, is_model))

    print(f"\n{N_MAPS} cartes jamais vues, {cfg.max_steps} pas maximum\n")
    print(f"{'agent':20}{'succès':>9}{'lave':>8}{'bloqué':>9}{'efficacité':>12}{'fidélité ws':>13}")
    for n, r in S.items():
        fid = f"{r['fidélité']:.0f} %" if r["fidélité"] is not None else "—"
        print(f"{n:20}{r['succès']:8.1f}%{r['lave']:7.1f}%{r['bloqué']:8.1f}%"
              f"{r['efficacité']:12.2f}{fid:>13}")
    print("\nSuccès selon la distance de l'objectif (plus court chemin) :")
    print(f"{'':20}" + "".join(f"{b:>7}" for b in BUCKETS))
    for n, r in S.items():
        print(f"{n:20}" + "".join(f"{r['par_distance'][b]:6.0f}%" for b in BUCKETS))
    counts = {b: 0 for b in BUCKETS}
    env = GridWorld(cfg)
    for m in range(N_MAPS):
        env.reset(seed=TEST_SEED * 10_000 + DEMO_OFFSET + m)
        counts[bucket(env.distance_to_goal())] += 1
    print(f"{'(nb de cartes)':20}" + "".join(f"{counts[b]:7d}" for b in BUCKETS))

    OUT.with_suffix(".json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    figure(S, OUT.with_suffix(".png"))
    print(f"\nFigure : {OUT.with_suffix('.png').name}")
