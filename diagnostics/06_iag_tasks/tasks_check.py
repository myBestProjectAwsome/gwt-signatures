"""
tasks_check.py — Mini-IAG, étape 4a : l'agent réalise-t-il des tâches variées ?

RÈGLE : ce diagnostic n'utilise QUE des tâches et des cartes publiques. Les
tâches et cartes du test final (evaluation/) ne sont jamais utilisées pendant
la construction de l'agent.

Tâches de développement :
  objectif          (tâche d'entraînement publique)
  clé               (tâche d'entraînement publique)
  objectif puis clé enchaînement de deux tâches publiques ; tâche de
                    développement choisie par l'auteur, hors du test
Cartes : monde v2 standard, graines de contrôle (1), 200 cartes par tâche.

Agents : aléatoire, oracle (plus court chemin exact), agent complet, et
ablations (un seul élément retiré à chaque fois) :
  sans surprise        ne retient pas les actions constatées inutiles
  avec critique        ajoute la critique (valeur au-delà de l'horizon), non retenue
  sans mémoire         ni lieux visités, ni actions inutiles
  sans workspace       exécute directement le plan classé premier
  modèles non entraînés  même câblage, poids aléatoires
Mesures : succès, lave, bloqué, efficacité, et succès selon la longueur du
plus court chemin.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mini_iag import Config  # noqa: E402
from mini_iag.architecture_v2 import ArchitectureV2  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.environment.keydoor_world import KeyDoorWorld  # noqa: E402
from mini_iag.task_agent import TaskAgent  # noqa: E402
from mini_iag.tasks import TRAINING_TASKS, Task, TaskTracker, optimal_steps, oracle_action  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402

N_MAPS, MAX_STEPS = 200, 40
DEV_TASKS = TRAINING_TASKS + (Task("objectif puis clé", ("goal", "key"),
                                   "Va sur l'objectif, PUIS ramasse la clé."),)
OUT = Path(__file__).with_name("tasks_results")
BUCKETS = ["1-3", "4-6", "7-9", "10+"]


def bucket(d):
    return "1-3" if d <= 3 else "4-6" if d <= 6 else "7-9" if d <= 9 else "10+"


class RandomAgent:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, task):
        pass

    def act(self, obs, world=None, tracker=None):
        return int(self.rng.integers(4))


class Oracle:
    def reset(self, task):
        self.task = task

    def act(self, obs, world=None, tracker=None):
        return oracle_action(world, self.task, tracker.progress)


def dev_map(cfg, task_index, i):
    world = KeyDoorWorld(cfg)
    for attempt in range(100):
        world.reset(seed=TEST_SEED * 10_000 + 3000 + task_index * 1000 + i * 7 + attempt)
        if optimal_steps(world, DEV_TASKS[task_index]):
            return world
    raise RuntimeError("carte insoluble")


def episode(agent, world, task):
    tracker, optimal = TaskTracker(task), optimal_steps(world, task)
    agent.reset(task)
    obs = world.observe()
    for t in range(1, MAX_STEPS + 1):
        action = agent.act(obs, world=world, tracker=tracker) if not isinstance(agent, TaskAgent) \
            else agent.act(obs)
        obs, events, dead = world.step(action)
        if tracker.update(events):
            return {"outcome": "succès", "steps": t, "optimal": optimal}
        if dead:
            return {"outcome": "lave", "steps": t, "optimal": optimal}
    return {"outcome": "bloqué", "steps": t, "optimal": optimal}


def evaluate(agent, cfg, n=N_MAPS):
    return {task.name: [episode(agent, dev_map(cfg, k, i), task) for i in range(n)]
            for k, task in enumerate(DEV_TASKS)}


def summary(res):
    succ = [r for r in res if r["outcome"] == "succès"]
    return {**{k: 100 * float(np.mean([r["outcome"] == k for r in res]))
               for k in ("succès", "lave", "bloqué")},
            "efficacité": float(np.mean([r["optimal"] / r["steps"] for r in succ])) if succ else 0.0,
            "par_distance": {b: (100 * float(np.mean([r["outcome"] == "succès" for r in res
                                                      if bucket(r["optimal"]) == b]))
                                 if any(bucket(r["optimal"]) == b for r in res) else None)
                             for b in BUCKETS}}


def figure(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    colors = {"succès": "#2a78d6", "lave": "#eb6834", "bloqué": "#b4b3aa"}
    fig, ax = plt.subplots(1, len(DEV_TASKS), figsize=(15, 4.8), sharey=True)
    names = list(S)[::-1]
    for a, task in zip(ax, DEV_TASKS):
        left = np.zeros(len(names))
        for k in ("succès", "lave", "bloqué"):
            vals = np.array([S[n][task.name][k] for n in names])
            a.barh(names, vals, left=left, color=colors[k], edgecolor="white", linewidth=2, label=k)
            for i, (v, l0) in enumerate(zip(vals, left)):
                if v >= 9:
                    a.text(l0 + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                           color="white" if k != "bloqué" else INK)
            left += vals
        a.set(title=f"« {task.name} »", xlim=(0, 100), xlabel="% des épisodes")
        a.spines[["top", "right"]].set_visible(False)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    fig.suptitle("Tâches publiques, 200 cartes de contrôle par tâche", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    if "--replot" in sys.argv:
        figure(json.loads(OUT.with_suffix(".json").read_text()), OUT.with_suffix(".png"))
        sys.exit()
    if not STEP4.exists():
        print("Pas de checkpoint de l'étape 4 : entraînement (python -m mini_iag.train_keydoor)...")
        train(verbose=False)
    cfg = Config.keydoor()
    full = load_task_agent()
    torch.manual_seed(0)
    fresh = ArchitectureV2(cfg)
    fresh.eval()
    agents = {
        "oracle": Oracle(),
        "complet": full,
        "sans surprise": load_task_agent(no_surprise=True),
        "avec critique": load_task_agent(cfg_overrides={"value_weight": 1.0}),
        "sans mémoire": load_task_agent(use_memory=False),
        "sans workspace": load_task_agent(use_workspace=False),
        "modèles non entraînés": TaskAgent(fresh, cfg, full.core.cue, full.core.sigma),
        "aléatoire": RandomAgent(),
    }
    S = {}
    for name, agent in agents.items():
        print(f"  évaluation : {name}...", flush=True)
        S[name] = {t: summary(r) for t, r in evaluate(agent, cfg).items()}

    print(f"\n{N_MAPS} cartes publiques de contrôle par tâche, {MAX_STEPS} pas maximum\n")
    for task in DEV_TASKS:
        print(f"Tâche « {task.name} »")
        print(f"  {'agent':24}{'succès':>9}{'lave':>8}{'bloqué':>9}{'efficacité':>12}   "
              + "  ".join(f"{b:>5}" for b in BUCKETS))
        for name in agents:
            r = S[name][task.name]
            dist = "  ".join(f"{v:4.0f}%" if v is not None else "    —" for v in r["par_distance"].values())
            print(f"  {name:24}{r['succès']:8.1f}%{r['lave']:7.1f}%{r['bloqué']:8.1f}%"
                  f"{r['efficacité']:12.2f}   {dist}")
        print()
    OUT.with_suffix(".json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    figure(S, OUT.with_suffix(".png"))
    print(f"Figure : {OUT.with_suffix('.png').name}")
