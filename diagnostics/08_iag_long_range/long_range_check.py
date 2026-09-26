"""
long_range_check.py — Mini-IAG : planifier plus loin.

Agents comparés, tâches publiques sur les cartes de contrôle (règle habituelle :
jamais les tâches ni les cartes du test final) :
  complet                 planification sur 5 pas imaginés + critique d'actions
                          (apprise sur l'expérience réelle) sur l'état réel
  sans critique           planification seule (l'agent des étapes 4a/4b)
  critique seule          aucune imagination : meilleure action selon la critique
  complet + vie           l'agent complet après N épisodes de vie continue
Références : oracle (plus court chemin), aléatoire.
Options : --episodes-vie N (défaut 1500 ; 0 pour sauter), --replot.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostics" / "06_iag_tasks"))

from tasks_check import BUCKETS, Oracle, RandomAgent, evaluate, summary  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.life import Life  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402

OUT = Path(__file__).with_name("long_range_results")
N_EVAL = 150


class CriticOnly:
    """Aucune imagination : l'action la mieux notée par la critique, sur l'état réel."""

    def __init__(self):
        self.agent = load_task_agent()
        self.cfg = self.agent.cfg

    def reset(self, task):
        self.agent.reset(task)

    @torch.no_grad()
    def act(self, obs, world=None, tracker=None):
        a = self.agent
        if a.prev is not None:                      # même suivi des sous-buts et des surprises
            before = a.tracker.progress
            if np.array_equal(a.prev, obs):
                a.useless.setdefault(obs.tobytes(), set()).add(a.last_action)
            a.tracker.update(a.detect(a.prev, obs))
            if a.tracker.progress != before and not a.tracker.done:
                a.arch.cost.configure(a.tracker.next_event)
        z = a.arch.world_model.encode(torch.as_tensor(obs)[None])
        q = a.arch.critic.q(z, a.arch.cost.target)[0].clone()
        for f in a.useless.get(obs.tobytes(), ()):
            q[f] = -1
        a.prev, a.last_action = obs, int(q.argmax())
        return a.last_action


def figure(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    tasks = list(next(iter(S.values())))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    names = list(S)[::-1]
    colors = {"succès": "#2a78d6", "lave": "#eb6834", "bloqué": "#b4b3aa"}
    a = ax[0]
    left = np.zeros(len(names))
    for k in ("succès", "lave", "bloqué"):
        vals = np.array([np.mean([S[n][t][k] for t in tasks]) for n in names])
        a.barh(names, vals, left=left, color=colors[k], edgecolor="white", linewidth=2, label=k)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v >= 7:
                a.text(l0 + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                       color="white" if k != "bloqué" else INK)
        left += vals
    a.set(title="Moyenne des trois tâches publiques", xlim=(0, 100), xlabel="% des épisodes")
    a.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    a = ax[1]
    col = {"sans critique": "#8a8980", "critique seule": "#1baf7a", "complet": "#2a78d6",
           "complet + vie": "#eb6834"}
    x = np.arange(len(BUCKETS))
    for n, c in col.items():
        if n not in S:
            continue
        y = []
        for b in BUCKETS:
            v = [S[n][t]["par_distance"][b] for t in tasks if S[n][t]["par_distance"][b] is not None]
            y.append(np.mean(v) if v else np.nan)
        a.plot(x, y, marker="o", color=c, lw=2, ms=8, markeredgecolor="white", markeredgewidth=2,
               label=n)
    a.set(title="Succès selon la distance de la cible", xticks=x, xticklabels=BUCKETS,
          xlabel="plus court chemin (pas)", ylabel="succès moyen (%)", ylim=(0, 105))
    a.legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    ax[1].grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-vie", type=int, default=1500)
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    if args.replot:
        figure(json.loads(OUT.with_suffix(".json").read_text()), OUT.with_suffix(".png"))
        sys.exit()
    if not STEP4.exists():
        train(verbose=False)
    cfg = Config.keydoor()
    agents = {"oracle": Oracle(), "complet": load_task_agent(),
              "sans critique": load_task_agent(cfg_overrides={"critic_weight": 0.0}),
              "critique seule": CriticOnly(), "aléatoire": RandomAgent()}
    S = {}
    for name, ag in agents.items():
        print(f"  évaluation : {name}...", flush=True)
        S[name] = {t: summary(r) for t, r in evaluate(ag, cfg, n=N_EVAL).items()}
    if args.episodes_vie:
        print(f"  vie de l'agent complet ({args.episodes_vie} épisodes)...", flush=True)
        ag = load_task_agent()
        life = Life(ag, path=Path("/tmp/long_range_life_unused.pt"))
        for i in range(args.episodes_vie):
            life.live_one()
            if (i + 1) % 500 == 0:
                print(f"    épisode {i + 1} : {life.recent(500)}", flush=True)
        S["complet + vie"] = {t: summary(r) for t, r in evaluate(ag, cfg, n=N_EVAL).items()}
    order = ["oracle", "complet + vie", "complet", "sans critique", "critique seule", "aléatoire"]
    S = {k: S[k] for k in order if k in S}
    print(f"\n{N_EVAL} cartes publiques de contrôle par tâche\n")
    for t in next(iter(S.values())):
        print(f"Tâche « {t} »")
        for n, r in S.items():
            d = "  ".join(f"{v:4.0f}%" if v is not None else "    —" for v in r[t]["par_distance"].values())
            print(f"  {n:18} succès {r[t]['succès']:5.1f}%  lave {r[t]['lave']:5.1f}%  "
                  f"bloqué {r[t]['bloqué']:5.1f}%   par distance {d}")
    OUT.with_suffix(".json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    figure(S, OUT.with_suffix(".png"))
    print(f"\nFigure : {OUT.with_suffix('.png').name}")
