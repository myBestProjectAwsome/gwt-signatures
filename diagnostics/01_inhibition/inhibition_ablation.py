"""
inhibition_ablation.py — Diagnostic du prototype `psyche`.

Question : qui pilote le choix du gagnant ?
  - la SAILLANCE (donc les jauges hormonales), comme voulu ?
  - ou l'INHIBITION DE RETOUR, qui impose une simple rotation ?

On lance la vraie boucle du package dans un monde SIMULÉ (FakeEnvironment),
pour que les résultats soient identiques sur toutes les machines.

Mesures :
  M1  accord saillance : % de cycles où le gagnant est la proposition
      de saillance brute la plus élevée (100 % = la saillance décide seule)
  M2  victoires "à vide" : % de victoires avec une saillance brute < 0.05
  M3  tour de rôle : % de cycles où le gagnant est le module présent
      qui a gagné le moins récemment (élevé = rotation mécanique)
  + répartition des gagnants par module
"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from psyche import FakeEnvironment, build, step  # noqa: E402

LEVELS = [0.0, 0.15, 0.5]
CYCLES = 1000
SEEDS = [0, 1, 2]
NAMES = ["capteur_systeme", "exploration", "social", "metacognition", "utilisateur"]


def simulate(inhibition, cycles, seed):
    random.seed(seed)
    env = FakeEnvironment(seed=seed)
    h, ws, user, modules = build(env)
    ws.inhibition = inhibition
    last_win = {n: -1 for n in NAMES}
    rows = []
    for t in range(cycles):
        winner, props = step(env, h, ws, user, modules)
        env.tick()
        if winner is None:
            continue
        best = max(props, key=lambda p: p.salience)
        present = [n for n in NAMES if n in {p.source for p in props}]
        oldest = min(present, key=lambda n: last_win[n])
        rows.append(dict(t=t, winner=winner.source, sal=winner.salience,
                         agree=winner.source == best.source,
                         oldest=winner.source == oldest,
                         E=h.energie, C=h.curiosite, A=h.attachement))
        last_win[winner.source] = t
    return rows


def summarize(rows):
    n = len(rows)
    return dict(
        agree=100 * sum(r["agree"] for r in rows) / n,
        empty=100 * sum(r["sal"] < 0.05 for r in rows) / n,
        rota=100 * sum(r["oldest"] for r in rows) / n,
        dist=Counter(r["winner"] for r in rows),
        n=n,
    )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, example = {}, {}
    for lv in LEVELS:
        runs = [simulate(lv, CYCLES, s) for s in SEEDS]
        example[lv] = runs[0]
        results[lv] = summarize([r for run in runs for r in run])

    print(f"{CYCLES} cycles x {len(SEEDS)} graines par niveau (environnement simulé)\n")
    print(f"{'inhibition':>10} | {'M1 accord':>9} | {'M2 à vide':>9} | {'M3 rotation':>11} | répartition")
    for lv, r in results.items():
        dist = "  ".join(f"{k[:5]}={100*r['dist'][k]/r['n']:4.1f}%" for k in NAMES)
        print(f"{lv:>10.2f} | {r['agree']:8.1f}% | {r['empty']:8.1f}% | {r['rota']:10.1f}% | {dist}")

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2)
    x = np.arange(len(LEVELS))

    a = fig.add_subplot(gs[0, 0])
    for i, (key, lab) in enumerate([("agree", "M1 accord saillance"),
                                    ("empty", "M2 victoires à vide"),
                                    ("rota", "M3 tour de rôle")]):
        a.bar(x + (i - 1) * 0.27, [results[lv][key] for lv in LEVELS], 0.27, label=lab)
    a.set_xticks(x, [str(lv) for lv in LEVELS])
    a.set(title="Qui décide ?", xlabel="inhibition", ylabel="% des cycles", ylim=(0, 105))
    a.legend()

    a = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(LEVELS))
    for k in NAMES:
        vals = np.array([100 * results[lv]["dist"][k] / results[lv]["n"] for lv in LEVELS])
        a.bar(x, vals, bottom=bottom, label=k)
        bottom += vals
    a.set_xticks(x, [str(lv) for lv in LEVELS])
    a.set(title="Répartition des gagnants", xlabel="inhibition", ylabel="%")
    a.legend(fontsize=8, loc="upper right")

    # Les jauges pilotent-elles le comportement ? (inhibition 0.15, 300 cycles)
    a = fig.add_subplot(gs[1, :])
    run = example[0.15][:300]
    ts = [r["t"] for r in run]
    for key, lab in [("E", "énergie"), ("C", "curiosité"), ("A", "attachement")]:
        a.plot(ts, [r[key] for r in run], label=lab)
    for name, y, c in [("exploration", -0.08, "tab:orange"),
                       ("utilisateur", -0.16, "tab:purple"),
                       ("social", -0.24, "tab:green")]:
        tw = [r["t"] for r in run if r["winner"] == name]
        a.scatter(tw, [y] * len(tw), marker="|", s=80, color=c, label=f"victoire {name}")
    a.set(title="Jauges et victoires (inhibition 0.15, 300 premiers cycles)",
          xlabel="cycle", ylim=(-0.3, 1.05))
    a.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    fig.tight_layout()
    out = Path(__file__).with_name("inhibition_results.png")
    fig.savefig(out, dpi=120)
    print(f"\nFigure : {out.name}")
