"""
life_check.py — Mini-IAG, étape 4b : vivre fait-il progresser ?

L'agent de l'étape 4a vit N épisodes (cartes publiques de la vie, tâches
publiques au hasard), en apprenant tous les 5 épisodes. Deux conditions :
  complet                    séances mêlant souvenirs récents et anciens (50/50)
  sans souvenirs anciens     séances sur l'expérience récente uniquement
                             (ablation : oubli catastrophique ?)

Mesures, avant et après la vie :
  - tâches publiques sur les cartes de CONTRÔLE (jamais vécues pendant la vie) ;
  - connaissances de physique sur l'expérience d'origine (agent aléatoire,
    graine de contrôle) : action retrouvée (S1) et anticipation des événements
    dans l'imagination (AUC à 1, 3 et 5 pas), dont la PORTE, que la vie ne
    demande jamais d'ouvrir.
RÈGLE : aucune tâche ni carte du test final n'est utilisée.
Options : --episodes N (défaut 1500), --replot.
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

from tasks_check import evaluate, summary  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.data_keydoor import EVENT_ORDER, collect_kd, collect_kd_segments  # noqa: E402
from mini_iag.life import Life  # noqa: E402
from mini_iag.metrics import action_identification, auc  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402

OUT = Path(__file__).with_name("life_results")
N_EVAL = 150


@torch.no_grad()
def physics(agent, trans, seg):
    wm, cost = agent.arch.world_model, agent.arch.cost
    s1 = action_identification(wm, trans.obs.float(), trans.action, trans.next_obs.float())[0]
    z0 = wm.encode(seg.obs.float())
    roll = wm.rollout(z0, seg.actions)
    prev = torch.cat([z0.unsqueeze(1), roll[:, :-1]], 1)
    out = {"S1": s1}
    for h in (0, 2, 4):
        m = seg.alive[:, h]
        p = cost.events(torch.cat([prev[m, h], roll[m, h]], -1))
        for i, e in enumerate(EVENT_ORDER):
            out[f"{e}@{h + 1}"] = auc(p[:, i], seg.events[m, h, i])
    return out


def run(cfg, episodes, old_fraction, trans, seg):
    agent = load_task_agent()
    life = Life(agent, old_fraction=old_fraction, path=Path("/tmp/life_check_unused.pt"))
    for i in range(episodes):
        life.live_one()
        if (i + 1) % 250 == 0:
            print(f"    épisode {i + 1} : {life.recent(250)}", flush=True)
    tasks = {t: summary(r) for t, r in evaluate(agent, cfg, n=N_EVAL).items()}
    return {"tasks": tasks, "physics": physics(agent, trans, seg),
            "history": [{k: r[k] for k in ("outcome", "surprise")} for r in life.history]}


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    col = {"avant": "#8a8980", "complet": "#2a78d6", "sans souvenirs anciens": "#eda100"}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    a = ax[0]
    w = 100
    for name in ("complet", "sans souvenirs anciens"):
        h = R[name]["history"]
        s = np.convolve([x["outcome"] == "succès" for x in h], np.ones(w) / w, "valid") * 100
        a.plot(np.arange(w, len(h) + 1), s, color=col[name], lw=2, label=name)
    a.axhline(np.mean([R["avant"]["tasks"][t]["succès"] for t in R["avant"]["tasks"]]),
              color=col["avant"], ls="--", lw=1.5, label="avant la vie (contrôle)")
    a.set(title=f"Succès pendant la vie (moyenne glissante sur {w} épisodes)",
          xlabel="épisode vécu", ylabel="succès (%)", ylim=(0, 100))
    a.legend(frameon=False, fontsize=9)
    tasks = list(R["avant"]["tasks"])
    x = np.arange(len(tasks))
    for j, name in enumerate(("avant", "complet", "sans souvenirs anciens")):
        vals = [R[name]["tasks"][t]["succès"] for t in tasks]
        bars = ax[1].bar(x + (j - 1) * 0.27, vals, 0.27, color=col[name], edgecolor="white",
                         linewidth=2, label=name)
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    ax[1].set_xticks(x, tasks, fontsize=9)
    ax[1].set(title="Tâches publiques, cartes de contrôle jamais vécues", ylabel="succès (%)",
              ylim=(0, 100))
    ax[1].legend(frameon=False, fontsize=8)
    keys = ["lava@5", "key@5", "door@5", "goal@5"]
    labels = ["lave", "clé", "porte", "objectif"]
    x = np.arange(len(keys))
    for j, name in enumerate(("avant", "complet", "sans souvenirs anciens")):
        ax[2].bar(x + (j - 1) * 0.27, [R[name]["physics"][k] for k in keys], 0.27,
                  color=col[name], edgecolor="white", linewidth=2, label=name)
    ax[2].set_xticks(x, labels)
    ax[2].set(title="Physique : anticipation à 5 pas imaginés", ylabel="AUC (0,5 = hasard)",
              ylim=(0.5, 1.0))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    if args.replot:
        figure(json.loads(OUT.with_suffix(".json").read_text()), OUT.with_suffix(".png"))
        sys.exit()
    if not STEP4.exists():
        train(verbose=False)
    cfg = Config.keydoor()
    trans = collect_kd(cfg, n_maps=300, steps=40, seed=TEST_SEED)
    seg = collect_kd_segments(cfg, n_maps=1500, per_map=6, horizon=5, seed=TEST_SEED)

    print("avant la vie...", flush=True)
    base = load_task_agent()
    R = {"avant": {"tasks": {t: summary(r) for t, r in evaluate(base, cfg, n=N_EVAL).items()},
                   "physics": physics(base, trans, seg)}}
    for name, old in (("complet", 0.5), ("sans souvenirs anciens", 0.0)):
        print(f"vie « {name} » ({args.episodes} épisodes)...", flush=True)
        R[name] = run(cfg, args.episodes, old, trans, seg)

    print(f"\nTâches publiques ({N_EVAL} cartes de contrôle, jamais vécues)")
    print(f"{'':26}" + "".join(f"{n:>26}" for n in R))
    for t in R["avant"]["tasks"]:
        print(f"  {t:24}" + "".join(
            f"{R[n]['tasks'][t]['succès']:9.1f}% (lave {R[n]['tasks'][t]['lave']:4.1f}%)" for n in R))
    print("\nPhysique, sur l'expérience d'origine (AUC à 1 / 3 / 5 pas imaginés)")
    print(f"  {'action retrouvée (S1)':24}" + "".join(f"{100 * R[n]['physics']['S1']:25.1f}%" for n in R))
    for e in EVENT_ORDER:
        print(f"  {e:24}" + "".join(
            f"{R[n]['physics'][e + '@1']:13.3f}/{R[n]['physics'][e + '@3']:.3f}/{R[n]['physics'][e + '@5']:.3f}"
            for n in R))
    OUT.with_suffix(".json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
    figure(R, OUT.with_suffix(".png"))
    print(f"\nFigure : {OUT.with_suffix('.png').name}")
