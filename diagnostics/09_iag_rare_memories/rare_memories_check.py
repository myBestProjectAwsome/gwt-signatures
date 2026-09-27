"""
rare_memories_check.py — Mini-IAG : la vie fait-elle encore oublier les portes ?

Constat de l'étape 4b : la vie ne demande jamais d'ouvrir une porte, et les
ouvertures de porte sont rares dans les souvenirs anciens (1,4 % des tirages).
Résultat : l'imagination des portes se dégradait en vivant.

Quatre conditions (agent complet, avec critique d'actions) :
  avant                   l'agent de l'étape 4a, sans vie
  vie, révision au hasard souvenirs anciens tirés uniformément
  vie, révision prioritaire  souvenirs anciens pondérés par leurs événements
                          rares (porte x11, clé x4, objectif x2)
  + exploration libre     révision prioritaire, et un quart des épisodes sans
                          tâche : l'agent se promène par curiosité
Chaque vie : N épisodes, tâches publiques uniquement (jamais de tâche porte).
On compte aussi les portes ouvertes et les pas « clé en main » vécus.

Mesures :
  - physique sur l'expérience d'origine (graine de contrôle) : action retrouvée,
    anticipation des événements à 1, 3 et 5 pas imaginés ;
  - PORTE FERMÉE : erreur de l'imagination quand l'agent pousse une porte sans
    avoir la clé (l'agent croit-il pouvoir passer ?) ;
  - tâches publiques sur les cartes de contrôle.
RÈGLE : aucune tâche ni carte du test final.
Options : --episodes N (défaut 1500 ; seul le passage complet met à jour
la figure et les données du README), --replot.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for d in ("06_iag_tasks", "07_iag_life"):
    sys.path.insert(0, str(ROOT / "diagnostics" / d))

from tasks_check import evaluate, summary  # noqa: E402
from life_check import physics  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.data_keydoor import EVENT_ORDER, collect_kd, collect_kd_segments  # noqa: E402
from mini_iag.environment.gridworld import MOVES  # noqa: E402
from mini_iag.environment.keydoor_world import AGENT, CARRY, DOOR  # noqa: E402
from mini_iag.life import Life  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402

OUT = Path(__file__).with_name("rare_memories_results")
N_EVAL = 150
CONDITIONS = {"vie, révision au hasard": dict(rare_bonus=None, explore_fraction=0.0, learn_critic=False),
              "vie, révision prioritaire": dict(rare_bonus="défaut", explore_fraction=0.0, learn_critic=False),
              "+ exploration libre": dict(rare_bonus="défaut", explore_fraction=0.25, learn_critic=False)}


@torch.no_grad()
def door_bump_error(agent, trans):
    """Erreur de prédiction sur les poussées de porte sans clé, et sur les autres pas."""
    wm = agent.arch.world_model
    o, n, a = trans.obs.float(), trans.next_obs.float(), trans.action
    cell = o[:, AGENT].flatten(1).argmax(1)
    r, c = cell // o.shape[-1], cell % o.shape[-1]
    dr = torch.tensor([MOVES[k][0] for k in a]); dc = torch.tensor([MOVES[k][1] for k in a])
    rows = torch.arange(len(a))
    bump = (o[rows, DOOR, r + dr, c + dc] > 0) & (o[:, CARRY].flatten(1).amax(1) == 0)
    err = ((wm.predict(wm.encode(o), a) - wm.target_encoder(n)) ** 2).sum(-1)
    return {"porte fermée": float(err[bump].mean()), "autres pas": float(err[~bump].mean()),
            "n_poussées": int(bump.sum())}


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    names = list(R)
    col = dict(zip(names, ["#8a8980", "#eda100", "#2a78d6", "#1baf7a"]))
    wd = 0.8 / len(names)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    keys, labels = ["door@1", "door@3", "door@5"], ["1 pas", "3 pas", "5 pas"]
    x = np.arange(3)
    for j, n in enumerate(names):
        ax[0].bar(x + (j - (len(names) - 1) / 2) * wd, [R[n]["physics"][k] for k in keys], wd, color=col[n],
                  edgecolor="white", linewidth=2, label=n)
    ax[0].set_xticks(x, labels)
    ax[0].set(title="Anticiper l'ouverture d'une porte (imagination)", ylabel="AUC", ylim=(0.8, 1.0))
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    vals = [R[n]["door_bump"]["porte fermée"] for n in names]
    b = ax[1].bar([n.replace(", ", ",\n") for n in names], vals, color=[col[n] for n in names], edgecolor="white", linewidth=2)
    for bb, v in zip(b, vals):
        ax[1].text(bb.get_x() + bb.get_width() / 2, v * 1.02, f"{v:.1f}", ha="center")
    ax[1].set(title="Erreur d'imagination quand il pousse\nune porte sans la clé (bas = mieux)")
    ax[1].tick_params(axis="x", labelsize=8)
    tasks = list(R[names[0]]["tasks"])
    x = np.arange(len(tasks))
    for j, n in enumerate(names):
        ax[2].bar(x + (j - (len(names) - 1) / 2) * wd, [R[n]["tasks"][t]["succès"] for t in tasks], wd,
                  color=col[n], edgecolor="white", linewidth=2, label=n)
    ax[2].set_xticks(x, tasks, fontsize=9)
    ax[2].set(title="Tâches publiques (cartes de contrôle)", ylabel="succès (%)", ylim=(0, 100))
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
    trans = collect_kd(cfg, n_maps=600, steps=40, seed=TEST_SEED)
    seg = collect_kd_segments(cfg, n_maps=1500, per_map=6, horizon=5, seed=TEST_SEED)

    def measure(agent):
        return {"physics": physics(agent, trans, seg), "door_bump": door_bump_error(agent, trans),
                "tasks": {t: summary(r) for t, r in evaluate(agent, cfg, n=N_EVAL).items()}}

    print("avant la vie...", flush=True)
    R = {"avant": measure(load_task_agent())}
    for name, kw in CONDITIONS.items():
        print(f"{name} ({args.episodes} épisodes)...", flush=True)
        agent = load_task_agent()
        life = Life(agent, path=Path("/tmp/rare_unused.pt"), **kw)
        lived = {"portes ouvertes": 0, "pas clé en main": 0, "pas": 0}
        for i in range(args.episodes):
            life.live_one()
            tr = life.buffer.episodes[-1]
            lived["portes ouvertes"] += int(tr["events"][:, EVENT_ORDER.index("door")].sum())
            lived["pas clé en main"] += int(tr["next_obs"][:, CARRY].reshape(len(tr["action"]), -1).max(1).sum())
            lived["pas"] += len(tr["action"])
            if (i + 1) % 500 == 0:
                print(f"    épisode {i + 1} : {life.recent(500)}", flush=True)
        R[name] = measure(agent) | {"vécu": lived}

    print("\nPhysique (expérience d'origine) : AUC à 1 / 3 / 5 pas imaginés")
    print(f"  {'':22}" + "".join(f"{n:>28}" for n in R))
    for e in EVENT_ORDER:
        print(f"  {e:22}" + "".join(f"{R[n]['physics'][e + '@1']:16.3f}/{R[n]['physics'][e + '@3']:.3f}/"
                                    f"{R[n]['physics'][e + '@5']:.3f}" for n in R))
    print(f"  {'action retrouvée (S1)':22}" + "".join(f"{100 * R[n]['physics']['S1']:27.1f}%" for n in R))
    print(f"\nErreur d'imagination, porte poussée sans clé ({R['avant']['door_bump']['n_poussées']} cas)")
    print(f"  {'porte fermée':22}" + "".join(f"{R[n]['door_bump']['porte fermée']:28.2f}" for n in R))
    print(f"  {'autres pas':22}" + "".join(f"{R[n]['door_bump']['autres pas']:28.2f}" for n in R))
    print("\nVécu pendant la vie")
    for k in ("portes ouvertes", "pas clé en main", "pas"):
        print(f"  {k:22}" + "".join(f"{R[n]['vécu'][k]:28d}" for n in R if "vécu" in R[n]))
    print("\nTâches publiques (150 cartes de contrôle)")
    for t in R["avant"]["tasks"]:
        print(f"  {t:22}" + "".join(
            f"{R[n]['tasks'][t]['succès']:16.1f}% (lave {R[n]['tasks'][t]['lave']:4.1f}%)" for n in R))
    if args.episodes == 1500:
        OUT.with_suffix(".json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        figure(R, OUT.with_suffix(".png"))
        print(f"\nFigure : {OUT.with_suffix('.png').name}")
    else:
        print("\n(passage partiel : figure et données du README non modifiées)")
