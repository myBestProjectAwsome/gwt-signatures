"""
mental_map_check.py — Mini-IAG : la carte mentale fait-elle planifier plus loin ?

Constat (diagnostics 8 et 10) : l'agent réussit presque tout à 1-3 pas, et
environ 20 % au-delà de 10 pas. L'imagination voit 5 pas devant ; la critique
d'actions devine mal plus loin.

La carte mentale (mini_iag/modules/mental_map.py) propage une valeur sur toutes
les cases de la carte, de proche en proche (itération sur les valeurs par
convolution). Elle est apprise par Q-learning hors ligne sur la même expérience
que la critique (l'agent qui marchait au hasard).

Agents :
  aléatoire, oracle (plus court chemin exact)
  complet (critique d'actions)   l'agent actuel : imagination + critique
  carte + imagination            la carte remplace la critique (poids 30 :
                                 la carte décide, l'imagination départage)
  carte seule                    poids 1000 : l'imagination ne compte presque plus
Cartes (publiques, graines de contrôle) :
  standard                       les cartes de toujours (2 murs intérieurs)
  murs nombreux                  7 murs intérieurs : chemins plus longs, détours.
                                 Famille publique ajoutée ici, jamais vue à
                                 l'entraînement (ni par la carte, ni par l'agent)
Mesures : succès, lave, succès selon la distance ; et, sur des états de cartes
de contrôle, la part d'actions optimales choisies par la carte et par la critique.
Aucune vie ici : tout est mesuré en zéro essai (aucune tâche n'a jamais été pratiquée).
RÈGLE : aucune tâche ni carte du test final.
Options : --n N (défaut 150 ; seul le passage complet met à jour la figure et
les données du README), --replot.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for d in ("06_iag_tasks", "10_iag_rehearsal"):
    sys.path.insert(0, str(ROOT / "diagnostics" / d))

from tasks_check import DEV_TASKS, Oracle, RandomAgent, bucket, dev_map, episode, summary  # noqa: E402
from rehearsal_check import distance  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.environment.keydoor_world import KeyDoorWorld  # noqa: E402
from mini_iag.modules.action_critic import TARGETS  # noqa: E402
from mini_iag.tasks import Task, optimal_steps  # noqa: E402
from mini_iag.tasks.search import _successors  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402
from mini_iag.train_map import MAP_PATH, train_map  # noqa: E402

OUT = Path(__file__).with_name("mental_map_results")
WALLS = 7
BUCKETS = ["1-3", "4-6", "7-9", "10+"]


def walls_map(cfg, k, i):
    world = KeyDoorWorld(replace(cfg, n_inner_walls=WALLS))
    for attempt in range(200):
        world.reset(seed=TEST_SEED * 10_000 + 40_000 + k * 1000 + i * 7 + attempt)
        if optimal_steps(world, DEV_TASKS[k]):
            return world
    raise RuntimeError("carte insoluble")


def run(agent, make_map):
    return {t.name: summary([episode(agent, make_map(k, i), t) for i in range(N)]) for k, t in enumerate(DEV_TASKS)}


def states(cfg, walls, n_maps, seed):
    rng = np.random.default_rng(seed)
    world, out = KeyDoorWorld(replace(cfg, n_inner_walls=walls)), []
    base = TEST_SEED * 10_000 + (60_000 if walls > 2 else 9000)
    for i in range(n_maps):
        world.reset(seed=base + i)
        dead = False
        for _ in range(int(rng.integers(0, 12))):
            _, _, dead = world.step(int(rng.integers(4)))
            if dead:
                break
        if dead:
            continue
        agent, has_key, keys, doors = world.state()
        start = (agent, has_key, keys, doors, 0)
        best = {}
        for target in TARGETS:
            task = Task(target, (target,), "")
            d = {a: distance(world, task, nxt) for a, nxt in _successors(world, start, task)}
            d = {a: v for a, v in d.items() if v is not None}
            if d:
                m = min(d.values())
                best[target] = ({a for a, v in d.items() if v == m}, m + 1)
        out.append((world.observe().copy(), best))
    return out


@torch.no_grad()
def choice_accuracy(q_fn, sts):
    res = {}
    for target in TARGETS:
        hits = {b: [] for b in BUCKETS}
        for obs, best in sts:
            if target in best:
                opt, dist = best[target]
                hits[bucket(dist)].append(int(q_fn(obs, target).argmax()) in opt)
        res[target] = {b: (100 * float(np.mean(v)) if v else None) for b, v in hits.items()} | {
            "n": {b: len(v) for b, v in hits.items()}}
    return res


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    names = [n for n in R["agents"] if n not in ("oracle",)]
    col = dict(zip(names, ["#b4b3aa", "#eda100", "#2a78d6", "#1baf7a"]))
    wd = 0.8 / len(names)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    tasks = [t.name for t in DEV_TASKS]
    for a, fam, title in ((ax[0], "standard", "Cartes standard"), (ax[1], "murs nombreux", "Cartes à murs nombreux (jamais vues)")):
        x = np.arange(len(tasks))
        for j, n in enumerate(names):
            a.bar(x + (j - (len(names) - 1) / 2) * wd, [R["agents"][n][fam][t]["succès"] for t in tasks], wd,
                  color=col[n], edgecolor="white", linewidth=2, label=n)
        a.axhline(75, color=INK, lw=1, ls="--")
        a.text(-0.45, 76.5, "seuil V1 (75 %)", fontsize=8)
        a.set_xticks(x, tasks, fontsize=9)
        a.set(title=title, ylabel="succès (%)", ylim=(0, 100))
    x = np.arange(len(BUCKETS))
    for j, n in enumerate(names):
        v = [np.nanmean([R["agents"][n][fam][t]["par_distance"][b] if R["agents"][n][fam][t]["par_distance"][b]
                         is not None else np.nan for fam in ("standard", "murs nombreux") for t in tasks])
             for b in BUCKETS]
        ax[2].bar(x + (j - (len(names) - 1) / 2) * wd, v, wd, color=col[n], edgecolor="white", linewidth=2)
    ax[2].set_xticks(x, [b + " pas" for b in BUCKETS])
    ax[2].set(title="Succès selon la distance (moyenne des 6 cas)", ylabel="succès (%)", ylim=(0, 100))
    ax[2].legend([plt.Rectangle((0, 0), 1, 1, color=col[n]) for n in names], names, frameon=False,
                 fontsize=8, loc="upper right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    if args.replot:
        figure(json.loads(OUT.with_suffix(".json").read_text()), OUT.with_suffix(".png"))
        sys.exit()
    N = args.n
    if not STEP4.exists():
        train(verbose=False)
    if not MAP_PATH.exists():
        train_map()
    cfg = Config.keydoor()
    maps = {"standard": lambda k, i: dev_map(cfg, k, i), "murs nombreux": lambda k, i: walls_map(cfg, k, i)}
    agents = {
        "aléatoire": lambda: RandomAgent(),
        "oracle": lambda: Oracle(),
        "complet (critique d'actions)": lambda: load_task_agent(),
        "carte + imagination": lambda: load_task_agent(mental_map=MAP_PATH, cfg_overrides={"map_weight": 30.0}),
        "carte seule": lambda: load_task_agent(mental_map=MAP_PATH, cfg_overrides={"map_weight": 1000.0}),
    }
    R = {"agents": {}, "choix": {}}
    for name, make in agents.items():
        print(f"{name}...", flush=True)
        R["agents"][name] = {fam: run(make(), m) for fam, m in maps.items()}

    ref = load_task_agent(mental_map=MAP_PATH)
    wm = ref.arch.world_model
    q_map = lambda obs, t: ref.arch.mental_map.q(torch.as_tensor(obs)[None], t)[0]
    q_crit = lambda obs, t: ref.arch.critic.q(wm.encode(torch.as_tensor(obs).float()[None]), t)[0]
    for fam, walls in (("standard", 2), ("murs nombreux", WALLS)):
        sts = states(cfg, walls, 1000, seed=0)
        R["choix"][fam] = {"critique d'actions": choice_accuracy(q_crit, sts), "carte mentale": choice_accuracy(q_map, sts)}

    for fam in maps:
        print(f"\nCartes {fam} ({N} par tâche), succès (lave)")
        print(f"  {'':30}" + "".join(f"{t.name:>24}" for t in DEV_TASKS))
        for n in agents:
            print(f"  {n:30}" + "".join(f"{R['agents'][n][fam][t.name]['succès']:12.1f}% (lave {R['agents'][n][fam][t.name]['lave']:4.1f}%)"
                                      for t in DEV_TASKS))
        print("  succès selon la distance (moyenne des 3 tâches) : " + " / ".join(BUCKETS))
        for n in agents:
            v = [np.nanmean([R['agents'][n][fam][t.name]['par_distance'][b] if R['agents'][n][fam][t.name]['par_distance'][b]
                             is not None else np.nan for t in DEV_TASKS]) for b in BUCKETS]
            print(f"  {n:30}" + " / ".join(f"{x:5.1f}" for x in v))
    print("\nAction optimale choisie (%) : " + " / ".join(BUCKETS))
    for fam in R["choix"]:
        for who, res in R["choix"][fam].items():
            for t in TARGETS:
                print(f"  {fam:14} {who:20} {t:5} " + " / ".join(
                    f"{res[t][b]:5.1f}" if res[t][b] is not None else "    -" for b in BUCKETS) +
                      f"   (n = {'/'.join(str(res[t]['n'][b]) for b in BUCKETS)})")
    held = "objectif puis clé"
    print("\nV1 de substitution (zéro essai ÷ oracle, seuil 0,75), tâche « objectif puis clé »")
    for n in agents:
        if n not in ("aléatoire", "oracle"):
            v = [R["agents"][n][fam][held]["succès"] / 100 for fam in maps]
            print(f"  {n:30} standard {v[0]:.2f}   murs nombreux {v[1]:.2f}")
    if N == 150:
        OUT.with_suffix(".json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        figure(R, OUT.with_suffix(".png"))
        print(f"\nFigure : {OUT.with_suffix('.png').name}")
    else:
        print("\n(passage partiel : figure et données du README non modifiées)")
