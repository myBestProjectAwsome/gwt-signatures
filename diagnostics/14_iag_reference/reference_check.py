"""
reference_check.py — Feuille de route v2, lot 1 : la référence reproductible.

Mesure la version du verdict (poids/verdict_2026-09-27) et ses témoins sur le
matériel de DÉVELOPPEMENT, avec un journal complet de chaque épisode (actions,
événements, issue, durée). C'est le point de départ auquel les lots suivants
seront comparés.

Agents :
  verdict (carte v2 + imagination)   l'agent du test du 27 septembre
  carte seule                        la carte décide seule (poids 1000)
  sans carte (critique d'actions)    l'agent d'avant la carte mentale
  sans workspace                     l'agent du verdict, plan classé premier exécuté directement
  aléatoire, oracle                  bornes (l'oracle triche : il lit l'état du monde ;
                                     il ignore la glace, donc absent de cette famille)
Matériel (public, graines de contrôle) :
  standard, murs nombreux : objectif, clé, objectif puis clé (sous-buts FOURNIS)
  glace (règle nouvelle, sans adaptation) : objectif, clé
Journal : reference_episodes.json.gz (un enregistrement par épisode).
"""
import argparse
import copy
import gzip
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for d in ("06_iag_tasks", "11_iag_learning_by_living", "12_iag_mental_map"):
    sys.path.insert(0, str(ROOT / "diagnostics" / d))

from tasks_check import DEV_TASKS, Oracle, RandomAgent, bucket, dev_map  # noqa: E402
from living_check import ice_maps  # noqa: E402
from mental_map_check import walls_map  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.data_keydoor import EVENT_ORDER  # noqa: E402
from mini_iag.reproduire import VERDICT, check  # noqa: E402
from mini_iag.task_agent import TaskAgent  # noqa: E402
from mini_iag.tasks import TRAINING_TASKS, TaskTracker, optimal_steps  # noqa: E402
from mini_iag.train_keydoor import load_task_agent  # noqa: E402

OUT = Path(__file__).with_name("reference_results.json")
LOG = Path(__file__).with_name("reference_episodes.json.gz")
MAX_STEPS = 40


def episode(agent, world, task):
    tracker, optimal = TaskTracker(task), optimal_steps(world, task)
    agent.reset(task)
    obs, actions, events, t0 = world.observe(), [], [], time.time()
    outcome = "bloqué"
    for t in range(1, MAX_STEPS + 1):
        a = agent.act(obs) if isinstance(agent, TaskAgent) else agent.act(obs, world=world, tracker=tracker)
        obs, ev, dead = world.step(a)
        actions.append(int(a)); events.append([EVENT_ORDER.index(e) for e in ev])
        if tracker.update(ev):
            outcome = "succès"
            break
        if dead:
            outcome = "lave"
            break
    return {"issue": outcome, "pas": t, "optimal": optimal, "actions": actions, "événements": events,
            "secondes": round(time.time() - t0, 3)}


def summarize(rows):
    ok = [r for r in rows if r["issue"] == "succès"]
    by = {}
    for b in ("1-3", "4-6", "7-9", "10+"):
        sel = [r for r in rows if r["optimal"] and bucket(r["optimal"]) == b]
        by[b] = 100 * float(np.mean([r["issue"] == "succès" for r in sel])) if sel else None
    return {"succès": 100 * float(np.mean([r["issue"] == "succès" for r in rows])),
            "lave": 100 * float(np.mean([r["issue"] == "lave" for r in rows])),
            "efficacité": float(np.mean([r["optimal"] / r["pas"] for r in ok])) if ok else 0.0,
            "par_distance": by, "n": len(rows)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    args = p.parse_args()
    N = args.n
    print("Poids du verdict :")
    if not check():
        sys.exit("poids du verdict modifiés")
    step4, mmap = VERDICT / "step4.pt", VERDICT / "mental_map.pt"
    cfg = Config.keydoor()
    agents = {
        "verdict (carte v2 + imagination)": lambda: load_task_agent(step4, mental_map=mmap),
        "carte seule": lambda: load_task_agent(step4, mental_map=mmap, cfg_overrides={"map_weight": 1000.0}),
        "sans carte (critique d'actions)": lambda: load_task_agent(step4),
        "sans workspace": lambda: load_task_agent(step4, mental_map=mmap, use_workspace=False),
        "aléatoire": lambda: RandomAgent(),
        "oracle": lambda: Oracle(),
    }
    ice = {t.name: ice_maps(cfg, t, N) for t in TRAINING_TASKS}
    families = {
        "standard": [(t, [dev_map(cfg, k, i) for i in range(N)]) for k, t in enumerate(DEV_TASKS)],
        "murs nombreux": [(t, [walls_map(cfg, k, i) for i in range(N)]) for k, t in enumerate(DEV_TASKS)],
        "glace": [(t, ice[t.name]) for t in TRAINING_TASKS],
    }
    R, log, t0 = {}, [], time.time()
    for name, make in agents.items():
        R[name] = {}
        for fam, cases in families.items():
            if fam == "glace" and name == "oracle":
                continue                   # l'oracle calcule avec la physique sans glace
            R[name][fam] = {}
            for task, worlds in cases:
                agent = make()
                rows = [episode(agent, copy.deepcopy(w), task) for w in worlds]
                for i, r in enumerate(rows):
                    log.append({"agent": name, "famille": fam, "tâche": task.name, "carte": i, **r})
                R[name][fam][task.name] = summarize(rows)
        print(f"[{time.time() - t0:5.0f} s] {name}", flush=True)

    for fam in families:
        print(f"\n{fam} ({N} cartes par tâche) : succès (lave)")
        tasks = list(R["aléatoire"][fam])
        print(f"  {'':34}" + "".join(f"{t:>24}" for t in tasks))
        for name in (n for n in agents if fam in R[n]):
            print(f"  {name:34}" + "".join(
                f"{R[name][fam][t]['succès']:12.1f}% (lave {R[name][fam][t]['lave']:4.1f}%)" for t in tasks))
    if N == 150:
        OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
        with gzip.open(LOG, "wt", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
        print(f"\nJournal : {LOG.name} ({len(log)} épisodes)")
