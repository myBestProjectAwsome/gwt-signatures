"""
inhibition_ablation.py — Diagnostic du prototype `psyche`.

Question : qui pilote le choix du gagnant ?
  - la SAILLANCE (donc les jauges hormonales), comme voulu ?
  - ou l'INHIBITION DE RETOUR, qui impose une simple rotation ?

On lance la vraie boucle du package (sans pause ni affichage) pour plusieurs
niveaux d'inhibition, et on mesure :
  M1  accord saillance : % de cycles où le gagnant est la proposition
      de saillance brute la plus élevée (100 % = la saillance décide seule)
  M2  victoires "à vide" : % de victoires avec une saillance brute < 0.05
      (un module qui ne veut rien obtient quand même l'attention)
  M3  prévisibilité : % de cycles où le gagnant est celui qui a gagné
      le moins récemment (proche de 100 % = tour de rôle mécanique)
  + répartition des gagnants par module
"""
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from psyche import Hormones, GlobalWorkspace                      # noqa: E402
from psyche.modules import SystemSensor, Explorer, Social, Metacognition  # noqa: E402

LEVELS = [0.0, 0.15, 0.5]
CYCLES = 1000
SEEDS = [0, 1, 2]


def simulate(inhibition, cycles, seed):
    random.seed(seed)
    h = Hormones()
    ws = GlobalWorkspace(inhibition=inhibition)
    modules = [SystemSensor(), Explorer(), Social(), Metacognition(ws)]
    names = [m.name for m in modules]
    last_win = {n: -1 for n in names}
    rows = []
    for t in range(cycles):
        props = [p for p in (m.propose(h) for m in modules) if p is not None]
        winner = ws.compete(props)
        if winner is None:
            continue
        best = max(props, key=lambda p: p.salience)
        # module présent ce cycle ayant gagné le moins récemment
        present = {p.source for p in props}
        oldest = min(present, key=lambda n: last_win[n])
        rows.append(dict(t=t, winner=winner.source, sal=winner.salience,
                         agree=winner.source == best.source,
                         oldest=winner.source == oldest,
                         cur=h.curiosite, att=h.attachement))
        last_win[winner.source] = t
        for m in modules:
            m.receive(winner)
        h.update(ws)
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

    results = {}
    example = {}
    for lv in LEVELS:
        runs = [simulate(lv, CYCLES, s) for s in SEEDS]
        example[lv] = runs[0]
        allrows = [r for run in runs for r in run]
        results[lv] = summarize(allrows)

    names = ["capteur_systeme", "exploration", "social", "metacognition"]
    print(f"{CYCLES} cycles x {len(SEEDS)} graines par niveau\n")
    print(f"{'inhibition':>10} | {'M1 accord':>9} | {'M2 à vide':>9} | {'M3 rotation':>11} | répartition")
    for lv, r in results.items():
        dist = "  ".join(f"{k[:5]}={100*r['dist'][k]/r['n']:4.1f}%" for k in names)
        print(f"{lv:>10.2f} | {r['agree']:8.1f}% | {r['empty']:8.1f}% | {r['rota']:10.1f}% | {dist}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    x = np.arange(len(LEVELS))
    for i, (key, lab) in enumerate([("agree", "M1 accord saillance"),
                                    ("empty", "M2 victoires à vide"),
                                    ("rota", "M3 tour de rôle")]):
        ax[0].bar(x + (i - 1) * 0.27, [results[lv][key] for lv in LEVELS], 0.27, label=lab)
    ax[0].set_xticks(x, [str(lv) for lv in LEVELS])
    ax[0].set(title="Qui décide ?", xlabel="inhibition", ylabel="% des cycles", ylim=(0, 105))
    ax[0].legend()

    bottom = np.zeros(len(LEVELS))
    for k in names:
        vals = np.array([100 * results[lv]["dist"][k] / results[lv]["n"] for lv in LEVELS])
        ax[1].bar(x, vals, bottom=bottom, label=k)
        bottom += vals
    ax[1].set_xticks(x, [str(lv) for lv in LEVELS])
    ax[1].set(title="Répartition des gagnants", xlabel="inhibition", ylabel="%")
    ax[1].legend(fontsize=8)

    run = example[0.5][:80]
    ypos = {n: i for i, n in enumerate(names)}
    ax[2].scatter([r["t"] for r in run], [ypos[r["winner"]] for r in run],
                  c=[r["sal"] for r in run], cmap="viridis", vmin=0, vmax=1, s=25)
    ax[2].set_yticks(range(len(names)), names)
    ax[2].set(title="80 premiers cycles, inhibition 0.5\n(couleur = saillance du gagnant)",
              xlabel="cycle")

    fig.tight_layout()
    out = Path(__file__).with_name("inhibition_results.png")
    fig.savefig(out, dpi=120)
    print(f"\nFigure : {out.name}")
