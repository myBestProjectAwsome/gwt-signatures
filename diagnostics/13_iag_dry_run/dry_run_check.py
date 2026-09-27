"""
dry_run_check.py — Mini-IAG, étape 5 : répétition à blanc de TOUTE la chaîne du test.

Le test final ne se passe qu'une fois. Avant, on fait tourner exactement la même
procédure (evaluation/run.py --test-final) sur du matériel PUBLIC, pour trouver
les bugs et avoir une idée honnête du résultat :

  agent final (build_final_agent : étape 4a + carte mentale, valeurs verrouillées)
  1. zéro essai sur tous les cas ;
  2. adaptation : 50 épisodes sur la tâche nouvelle (cartes d'adaptation publiques,
     distinctes des cartes de mesure), puis mesure ;
  3. référence sans connaissances (build_relearner) : même adaptation, même mesure ;
  4. référence à nouveau après adaptation (oubli), contrôle du verrou (check_locked) ;
  5. règles V1 à V5 avec les MÊMES seuils et les mêmes fonctions (evaluation/battery.py).

Cas publics (aucune tâche ni carte du test) :
  P1  « objectif puis clé » (jamais pratiquée), cartes standard      -> V1, V2
  PW  « objectif », cartes à murs nombreux (jamais vues)              -> V3
  PR  « objectif », cartes standard (référence)                       -> V3, V5
Humain remplacé par l'oracle (100 %) : c'est pessimiste pour l'agent.
Il n'y a qu'un monde nouveau public (PW) : V3 est vérifiée sur lui seul.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostics" / "06_iag_tasks"))

from tasks_check import DEV_TASKS  # noqa: E402
from evaluation.battery import (HUMAN_THRESHOLD, PRACTICE_EPISODES, RETENTION, WORLD_FLOOR,  # noqa: E402
                                WORLD_ROBUSTNESS, bootstrap_diff, check_locked, success_rate,
                                summarize)
from mini_iag import Config  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.environment.keydoor_world import KeyDoorWorld  # noqa: E402
from mini_iag.final_agent import build_final_agent, build_relearner  # noqa: E402
from mini_iag.tasks import TaskTracker, optimal_steps  # noqa: E402

OUT = Path(__file__).with_name("dry_run_results.json")
GOAL, HELD = DEV_TASKS[0], DEV_TASKS[2]


@dataclass(frozen=True)
class PublicCase:
    code: str
    task: object
    walls: int
    max_steps: int = 40


CASES = (PublicCase("P1", HELD, 2), PublicCase("PW", GOAL, 7), PublicCase("PR", GOAL, 2))


def make_map(cfg, case, i, base):
    world = KeyDoorWorld(replace(cfg, n_inner_walls=case.walls))
    k = [c.code for c in CASES].index(case.code)
    for attempt in range(200):
        world.reset(seed=base + k * 10_000 + i * 7 + attempt)
        if optimal_steps(world, case.task):
            return world
    raise RuntimeError("carte insoluble")


MEASURE = TEST_SEED * 10_000 + 100_000          # cartes de mesure (contrôle)
PRACTICE = TEST_SEED * 10_000 + 200_000         # cartes d'adaptation, distinctes


def run_case(agent, cfg, case, n):
    out = []
    for i in range(n):
        world = make_map(cfg, case, i, MEASURE)
        tracker, optimal = TaskTracker(case.task), optimal_steps(world, case.task)
        agent.reset(case.task)
        obs, res = world.observe(), None
        for t in range(1, case.max_steps + 1):
            obs, events, dead = world.step(agent.act(obs))
            if tracker.update(events):
                res = {"success": True, "steps": t, "optimal": optimal, "lava": False}
                break
            if dead:
                break
        out.append(res or {"success": False, "steps": t, "optimal": optimal, "lava": bool(dead)})
    return out


def practice(agent, cfg, case):
    agent.practice(lambda i: make_map(cfg, case, i, PRACTICE), case.task, PRACTICE_EPISODES)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    args = p.parse_args()
    n, cfg, t0 = args.n, Config(), time.time()
    say = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)
    by = {c.code: c for c in CASES}

    agent = build_final_agent()
    fp = agent.cost_module.fingerprint()
    say(f"agent final, valeurs verrouillées ({fp[:16]}...)")
    zero = {c.code: run_case(agent, cfg, c, n) for c in CASES}
    for k, r in zero.items():
        say(f"zéro essai  {k}  {summarize(r)['succès']:5.1f} % (lave {summarize(r)['lave']:.1f} %)")
    practice(agent, cfg, by["P1"])
    adapted = {"P1": run_case(agent, cfg, by["P1"], n)}
    say(f"adapté      P1  {summarize(adapted['P1'])['succès']:5.1f} %")
    after = success_rate(run_case(agent, cfg, by["PR"], n))
    say(f"PR après adaptation {100 * after:5.1f} %")
    locked = check_locked(agent.cost_module)
    relearner = build_relearner(seed=0)
    practice(relearner, cfg, by["P1"])
    relearned = {"P1": run_case(relearner, cfg, by["P1"], n)}
    say(f"référence   P1  {summarize(relearned['P1'])['succès']:5.1f} %")

    # mêmes règles, mêmes seuils (humain remplacé par l'oracle : 100 %)
    r1 = min(success_rate(zero["P1"]) / 1.0, 1.5)
    d, lo, hi = bootstrap_diff(adapted["P1"], relearned["P1"])
    ref = success_rate(zero["PR"])
    w = success_rate(zero["PW"])
    V = {"V1": {"rapport": r1, "réussi": r1 >= HUMAN_THRESHOLD},
         "V2": {"différence": d, "IC95": [lo, hi], "réussi": lo > 0},
         "V3": {"rapport": w / ref if ref else 0, "succès": w,
                "réussi": ref > 0 and w / ref >= WORLD_ROBUSTNESS and w >= WORLD_FLOOR},
         "V4": {"contrôles": locked, "réussi": all(locked.values()) and agent.cost_module.fingerprint() == fp},
         "V5": {"rapport": after / ref if ref else 0, "réussi": ref > 0 and after / ref >= RETENTION}}
    print("\nRépétition à blanc (matériel public, humain = oracle)")
    for k, v in V.items():
        detail = {kk: vv for kk, vv in v.items() if kk != "réussi"}
        print(f"  {k}  {'réussi' if v['réussi'] else 'RATÉ  '}  {detail}")
    print(f"  => {'toutes les règles passent' if all(v['réussi'] for v in V.values()) else 'au moins une règle ratée'}")
    print(f"\nDurée : {time.time() - t0:.0f} s")
    if n == 150:
        OUT.write_text(json.dumps({"zéro_essai": {k: summarize(r) for k, r in zero.items()},
                                   "adapté": summarize(adapted["P1"]),
                                   "référence": summarize(relearned["P1"]),
                                   "PR_après": 100 * after, "règles": V},
                                  indent=1, ensure_ascii=False, default=float))
