"""
living_check.py — Mini-IAG : apprend-elle vraiment EN VIVANT ?

Le diagnostic 10 a montré que « l'agent progresse après sa vie » ne prouve pas
qu'il apprend de ce qu'il vit : deux séances d'apprentissage sur ses seuls
souvenirs anciens donnent presque le même progrès. Deux corrections :

1. UN TÉMOIN. La même quantité d'apprentissage (300 séances), SANS rien vivre.
   L'apprentissage en vivant n'est prouvé que si la vraie vie fait mieux.
2. QUELQUE CHOSE DE NOUVEAU À APPRENDRE. Dans le monde d'origine, l'agent
   connaissait déjà toute la physique. On ajoute une règle absente de toute
   son expérience : la GLACE (une case qui fait glisser jusqu'au prochain
   obstacle). Sa perception gagne un sens (8e canal, poids partant de zéro :
   rien ne change pour ce qu'elle percevait déjà).

Conditions (même agent de départ, avec le nouveau sens) :
  avant                       aucun apprentissage
  témoin : séances sans vivre 300 séances sur les seuls souvenirs anciens
  vie normale                 1 500 épisodes, monde sans glace
  vie avec glace              1 500 épisodes, monde avec 5 cases de glace
Mesures :
  - tâches « objectif » et « clé » dans le monde AVEC glace (cartes de contrôle) ;
  - les trois tâches publiques dans le monde normal (ce qui est gardé) ;
  - physique de la glace : après une glissade réelle, l'imagination place-t-elle
    l'agent dans la bonne case ? anticipe-t-elle la mort par glissade dans la lave ?
La glace n'existe ni dans l'expérience d'origine, ni dans le test.
RÈGLE : aucune tâche ni carte du test final.
Options : --episodes N (défaut 1500 ; seul le passage complet met à jour la
figure et les données du README), --replot.
"""
import argparse
import copy
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
from mini_iag.life import ContinualLearner, ExperienceBuffer, Life, anchor_segments  # noqa: E402
from mini_iag.life.episode import run_episode  # noqa: E402
from mini_iag.metrics import agent_cell, apply_linear_probe, auc, fit_linear_probe  # noqa: E402
from mini_iag.tasks import TRAINING_TASKS  # noqa: E402
from mini_iag.train_keydoor import STEP4, load_task_agent, train  # noqa: E402

OUT = Path(__file__).with_name("living_results")
N_ICE, N_EVAL, MAX_STEPS = 5, 150, 40
ICE_BASE = TEST_SEED * 10_000 + 20_000          # cartes de contrôle, jamais vécues


def ice_optimal(world, task):
    """Plus court chemin avec la physique de la glace (simulation), ou None."""
    def key(w, p):
        return w.state() + (p,)
    start = copy.deepcopy(world)
    seen, q = {key(start, 0)}, deque([(start, 0, 0)])
    while q:
        w, p, d = q.popleft()
        for a in range(4):
            w2 = copy.deepcopy(w)
            _, events, dead = w2.step(a)
            if dead:
                continue
            p2 = p
            for e in events:
                if p2 < len(task.sequence) and e == task.sequence[p2]:
                    p2 += 1
            if p2 == len(task.sequence):
                return d + 1
            k = key(w2, p2)
            if k not in seen and d < 30:
                seen.add(k)
                q.append((w2, p2, d + 1))
    return None


def ice_maps(cfg, task, n):
    maps, i = [], 0
    while len(maps) < n:
        w = KeyDoorWorld(cfg, n_ice=N_ICE)
        w.reset(seed=ICE_BASE + 1000 * TRAINING_TASKS.index(task) + i)
        i += 1
        if ice_optimal(w, task) is not None:
            maps.append(w)
    return maps


def ice_tasks(agent, maps):
    res = {}
    for task, worlds in maps.items():
        out = [run_episode(agent, copy.deepcopy(w), task, MAX_STEPS)[0] for w in worlds]
        res[task.name] = {k: 100 * float(np.mean([o == k for o in out])) for k in ("succès", "lave", "bloqué")}
    return res


def ice_transitions(cfg, n_maps=400, steps=30, seed=1):
    """Transitions réelles au hasard dans le monde avec glace (cartes de contrôle)."""
    rng = np.random.default_rng(seed)
    o, a, n, slide, lava = [], [], [], [], []
    for i in range(n_maps):
        w = KeyDoorWorld(cfg, n_ice=N_ICE)
        obs = w.reset(seed=ICE_BASE + 50_000 + i)
        for _ in range(steps):
            act = int(rng.integers(4))
            before = w.agent
            nxt, events, dead = w.step(act)
            o.append(obs); a.append(act); n.append(nxt)
            slide.append(abs(w.agent[0] - before[0]) + abs(w.agent[1] - before[1]) > 1)
            lava.append(dead)
            if dead:
                break
            obs = nxt
    t = lambda x: torch.tensor(np.array(x))
    return t(o).float(), t(a), t(n).float(), t(slide), t(lava)


@torch.no_grad()
def ice_physics(agent, trans):
    """Où l'imagination place-t-elle l'agent après l'action ? Une sonde linéaire,
    apprise sur les états RÉELS (70 % des transitions), lit la case de l'agent ;
    on l'applique aux états IMAGINÉS des 30 % restants."""
    wm, cost = agent.arch.world_model, agent.arch.cost
    o, a, n, slide, lava = trans
    z, zn = wm.encode(o), wm.target_encoder(n)
    pred = wm.predict(z, a)
    cell, n_cells = agent_cell(n)
    k = int(0.7 * len(o))
    W = fit_linear_probe(zn[:k], cell[:k], n_cells)
    real_ok = apply_linear_probe(W, zn[k:]) == cell[k:]
    imag_ok = apply_linear_probe(W, pred[k:]) == cell[k:]
    sl = slide[k:]
    danger = cost.evaluate(z, pred)[0]
    slide_lava = lava & slide
    return {"case imaginée, glissades": 100 * float(imag_ok[sl].float().mean()),
            "case imaginée, autres pas": 100 * float(imag_ok[~sl].float().mean()),
            "case réelle lue (sonde)": 100 * float(real_ok.float().mean()),
            "lave par glissade (AUC)": auc(danger[slide], slide_lava[slide].float()),
            "n glissades": int(slide.sum()), "n morts par glissade": int(slide_lava.sum())}


def figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    names = list(R)
    col = dict(zip(names, ["#8a8980", "#b4b3aa", "#2a78d6", "#1baf7a"]))
    wd = 0.8 / len(names)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

    def bars(a, groups, get, title, ylabel, ylim):
        x = np.arange(len(groups))
        for j, nm in enumerate(names):
            a.bar(x + (j - (len(names) - 1) / 2) * wd, [get(nm, g) for g in groups], wd,
                  color=col[nm], edgecolor="white", linewidth=2, label=nm)
        a.set_xticks(x, groups, fontsize=9)
        a.set(title=title, ylabel=ylabel, ylim=ylim)

    bars(ax[0], list(R[names[0]]["glace"]), lambda nm, g: R[nm]["glace"][g]["succès"],
         "Monde AVEC glace (règle nouvelle)", "succès (%)", (0, 100))
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    bars(ax[1], ["case imaginée, glissades", "case imaginée, autres pas"],
         lambda nm, g: R[nm]["physique glace"][g],
         "Après l'action, l'imagination place-t-elle\nl'agent dans la bonne case ?", "%", (0, 100))
    bars(ax[2], list(R[names[0]]["normal"]), lambda nm, g: R[nm]["normal"][g]["succès"],
         "Monde normal (ce qui est gardé)", "succès (%)", (0, 100))
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
    maps = {t: ice_maps(cfg, t, N_EVAL) for t in TRAINING_TASKS}
    trans = ice_transitions(cfg)
    print(f"{int(trans[3].sum())} glissades réelles pour mesurer la physique de la glace", flush=True)
    n_sessions = args.episodes // 5

    def fresh():
        agent = load_task_agent()
        agent.arch.world_model.grow_senses(1)
        return agent

    def measure(agent):
        return {"glace": ice_tasks(agent, maps), "physique glace": ice_physics(agent, trans),
                "normal": {t: summary(r) for t, r in evaluate(agent, cfg, n=N_EVAL).items()}}

    print("avant...", flush=True)
    R = {"avant": measure(fresh())}
    print(f"témoin : {n_sessions} séances sans vivre...", flush=True)
    agent = fresh()
    learner = ContinualLearner(agent.arch, ExperienceBuffer(anchor_segments(cfg), horizon=cfg.planning_horizon),
                               learn_critic=True)
    for _ in range(n_sessions):
        learner.update()
    R["témoin : séances sans vivre"] = measure(agent)
    for name, n_ice in (("vie normale", 0), ("vie avec glace", N_ICE)):
        print(f"{name} ({args.episodes} épisodes)...", flush=True)
        agent = fresh()
        life = Life(agent, path=Path("/tmp/living_unused.pt"), n_ice=n_ice)
        for i in range(args.episodes):
            life.live_one()
            if (i + 1) % 500 == 0:
                print(f"    épisode {i + 1} : {life.recent(500)}", flush=True)
        R[name] = measure(agent)

    print(f"\nMonde AVEC glace ({N_EVAL} cartes de contrôle), succès (lave)")
    print(f"  {'':14}" + "".join(f"{n:>30}" for n in R))
    for t in R["avant"]["glace"]:
        print(f"  {t:14}" + "".join(f"{R[n]['glace'][t]['succès']:18.1f}% (lave {R[n]['glace'][t]['lave']:4.1f}%)"
                                    for n in R))
    print("\nPhysique de la glace")
    for k in ("case imaginée, glissades", "case imaginée, autres pas", "case réelle lue (sonde)",
              "lave par glissade (AUC)"):
        print(f"  {k:26}" + "".join(f"{R[n]['physique glace'][k]:26.3f}" for n in R))
    print(f"  ({R['avant']['physique glace']['n glissades']} glissades, "
          f"{R['avant']['physique glace']['n morts par glissade']} morts par glissade)")
    print(f"\nMonde normal ({N_EVAL} cartes de contrôle), succès (lave)")
    for t in R["avant"]["normal"]:
        print(f"  {t:18}" + "".join(f"{R[n]['normal'][t]['succès']:16.1f}% (lave {R[n]['normal'][t]['lave']:4.1f}%)"
                                    for n in R))
    if args.episodes == 1500:
        OUT.with_suffix(".json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        figure(R, OUT.with_suffix(".png"))
        print(f"\nFigure : {OUT.with_suffix('.png').name}")
    else:
        print("\n(passage partiel : figure et données du README non modifiées)")
