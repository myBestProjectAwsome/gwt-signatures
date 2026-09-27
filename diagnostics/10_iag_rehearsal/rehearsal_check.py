"""
rehearsal_check.py — Mini-IAG : répétition générale du test final (test blanc).

Le test final mesure surtout une chose (règle V1) : réussir, du PREMIER coup,
une tâche jamais pratiquée. Jusqu'ici, aucun diagnostic ne le mesurait :
la vie pratiquait toutes les tâches publiques.

Test blanc, sur du matériel public uniquement :
  - la tâche composée « objectif puis clé » est RETIRÉE de la vie ;
  - après la vie, on mesure l'agent sur cette tâche en zéro essai ;
  - « humain » de substitution : l'oracle (100 % sur ces cartes). C'est une
    hypothèse optimiste pour l'humain, donc pessimiste pour l'agent ;
  - règle V1 de substitution : succès zéro essai ÷ 100 % ≥ 0,75.

Conditions (agent complet, 1 500 épisodes de vie chacune) :
  avant la vie
  vie sans la tâche                 vie sur « objectif » et « clé » seulement
  vie sans la tâche + critique      idem, et la critique d'actions apprend en vivant
  vie avec la tâche (pratiquée)     référence : ce que la pratique apporte

Mesure supplémentaire : la critique choisit-elle une action optimale ?
Sur des états de cartes de contrôle, pour chaque événement visé (objectif,
clé, porte), part des cas où l'action préférée de la critique est l'une des
premières actions d'un plus court chemin (calculé exactement). La porte est
mesurée comme connaissance du monde, comme au diagnostic 9 : aucune tâche
« porte » n'est jamais pratiquée.

RÈGLE : aucune tâche ni carte du test final.
Options : --episodes N (défaut 1500 ; seul le passage complet met à jour la
figure et les données du README), --replot.
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostics" / "06_iag_tasks"))

from tasks_check import evaluate, summary  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.environment.keydoor_world import KeyDoorWorld  # noqa: E402
from mini_iag.life import Life  # noqa: E402
from mini_iag.life.life import LIFE_TASKS  # noqa: E402
from mini_iag.modules.action_critic import TARGETS  # noqa: E402
from mini_iag.tasks import Task  # noqa: E402
from mini_iag.tasks.search import _successors  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402

OUT = Path(__file__).with_name("rehearsal_results")
N_EVAL, HELD = 150, "objectif puis clé"
WITHOUT = tuple(t for t in LIFE_TASKS if t.name != HELD)
CONDITIONS = {
    "vie sans la tâche": dict(tasks=WITHOUT, learn_critic=False),
    "vie sans la tâche + critique": dict(tasks=WITHOUT, learn_critic=True),
    "vie avec la tâche (pratiquée)": dict(tasks=LIFE_TASKS, learn_critic=False),
}


def distance(world, task, start):
    """Plus court chemin (en pas) depuis un état de recherche, ou None."""
    seen, q = {start}, deque([(start, 0)])
    while q:
        s, d = q.popleft()
        if s[4] == len(task.sequence):
            return d
        for _, nxt in _successors(world, s, task):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return None


def critic_states(cfg, n_maps=250, seed=0):
    """États de cartes de contrôle, avec les actions optimales pour chaque événement."""
    rng = np.random.default_rng(seed)
    world, out = KeyDoorWorld(cfg), []
    for i in range(n_maps):
        world.reset(seed=TEST_SEED * 10_000 + 9000 + i)
        for _ in range(int(rng.integers(0, 12))):             # quelques pas au hasard
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
def critic_accuracy(agent, states):
    wm, critic = agent.arch.world_model, agent.arch.critic
    res = {}
    for target in TARGETS:
        hits = {"1-3": [], "4-6": [], "7+": []}
        for obs, best in states:
            if target not in best:
                continue
            opt, dist = best[target]
            a = int(critic.q(wm.encode(torch.as_tensor(obs).float()[None]), target)[0].argmax())
            hits["1-3" if dist <= 3 else "4-6" if dist <= 6 else "7+"].append(a in opt)
        res[target] = {k: (100 * float(np.mean(v)) if v else None) for k, v in hits.items()} | {
            "tous": 100 * float(np.mean(sum(hits.values(), [])))}
    return res


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    names = list(R)
    col = dict(zip(names, ["#8a8980", "#eda100", "#1baf7a", "#2a78d6"]))
    wd = 0.8 / len(names)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    tasks = list(R[names[0]]["tasks"])
    x = np.arange(len(tasks))
    for j, n in enumerate(names):
        ax[0].bar(x + (j - (len(names) - 1) / 2) * wd, [R[n]["tasks"][t]["succès"] for t in tasks], wd,
                  color=col[n], edgecolor="white", linewidth=2, label=n)
    ax[0].axhline(75, color=INK, lw=1, ls="--")
    ax[0].text(len(tasks) - 0.5, 76.5, "seuil V1 (75 %)", ha="right", fontsize=8)
    ax[0].set_xticks(x, [t + ("\n(jamais pratiquée*)" if t == HELD else "") for t in tasks], fontsize=9)
    ax[0].set(title="Tâches publiques (cartes de contrôle)", ylabel="succès (%)", ylim=(0, 100))
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    buckets = ["1-3", "4-6", "7-9", "10+"]
    x = np.arange(len(buckets))
    for j, n in enumerate(names):
        v = [R[n]["tasks"][HELD]["par_distance"][b] or 0 for b in buckets]
        ax[1].bar(x + (j - (len(names) - 1) / 2) * wd, v, wd, color=col[n], edgecolor="white", linewidth=2)
    ax[1].set_xticks(x, [b + " pas" for b in buckets])
    ax[1].set(title=f"« {HELD} » selon la distance (plus court chemin)", ylabel="succès (%)",
              ylim=(0, 100))
    x = np.arange(len(TARGETS))
    for j, n in enumerate(names):
        ax[2].bar(x + (j - (len(names) - 1) / 2) * wd, [R[n]["critic"][t]["tous"] for t in TARGETS], wd,
                  color=col[n], edgecolor="white", linewidth=2)
    ax[2].set_xticks(x, ["objectif", "clé", "porte"])
    ax[2].set(title="La critique choisit une action optimale", ylabel="%", ylim=(0, 100))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID)
    fig.text(0.01, 0.01, "* sauf pour « vie avec la tâche », où elle est pratiquée (référence)",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
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
    states = critic_states(cfg)

    def measure(agent):
        return {"tasks": {t: summary(r) for t, r in evaluate(agent, cfg, n=N_EVAL).items()},
                "critic": critic_accuracy(agent, states)}

    print(f"{len(states)} états pour mesurer la critique", flush=True)
    print("avant la vie...", flush=True)
    R = {"avant la vie": measure(load_task_agent())}
    for name, kw in CONDITIONS.items():
        print(f"{name} ({args.episodes} épisodes)...", flush=True)
        agent = load_task_agent()
        life = Life(agent, path=Path("/tmp/rehearsal_unused.pt"), **kw)
        assert kw["tasks"] == LIFE_TASKS or all(t.name != HELD for t in life.tasks)
        for i in range(args.episodes):
            life.live_one()
            if (i + 1) % 500 == 0:
                print(f"    épisode {i + 1} : {life.recent(500)}", flush=True)
        R[name] = measure(agent)

    print(f"\nTâches publiques ({N_EVAL} cartes de contrôle), succès (lave)")
    print(f"  {'':22}" + "".join(f"{n:>32}" for n in R))
    for t in R["avant la vie"]["tasks"]:
        print(f"  {t:22}" + "".join(
            f"{R[n]['tasks'][t]['succès']:20.1f}% (lave {R[n]['tasks'][t]['lave']:4.1f}%)" for n in R))
    print(f"\n« {HELD} » selon la distance")
    for b in ("1-3", "4-6", "7-9", "10+"):
        print(f"  {b:22}" + "".join(
            f"{(R[n]['tasks'][HELD]['par_distance'][b] or 0):31.1f}%" for n in R))
    print("\nV1 de substitution (zéro essai ÷ oracle, seuil 0,75)")
    for n in ("vie sans la tâche", "vie sans la tâche + critique"):
        if n in R:
            v = R[n]["tasks"][HELD]["succès"] / 100
            print(f"  {n:32} {v:.2f}  {'réussi' if v >= 0.75 else 'raté'}")
    print("\nLa critique choisit une action optimale (%) : tous / 1-3 / 4-6 / 7+ pas")
    for t in TARGETS:
        print(f"  {t:22}" + "".join(
            f"{R[n]['critic'][t]['tous']:13.1f} / " + " / ".join(
                f"{R[n]['critic'][t][b]:.0f}" if R[n]['critic'][t][b] is not None else "-"
                for b in ("1-3", "4-6", "7+")) for n in R))
    if args.episodes == 1500:
        OUT.with_suffix(".json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        figure(R, OUT.with_suffix(".png"))
        print(f"\nFigure : {OUT.with_suffix('.png').name}")
    else:
        print("\n(passage partiel : figure et données du README non modifiées)")
