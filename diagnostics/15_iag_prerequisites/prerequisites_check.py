"""
prerequisites_check.py — Feuille de route v2, lot 2 : découvrir les prérequis.

On ne demande QUE l'objectif. Quatre situations (mini_iag/environment/situations.py),
chaque carte vérifiée par l'oracle :
  accessible       aller directement au but
  incontournable   prendre la clé, puis franchir la porte
  ouverte          ignorer la clé inutile
  detour           prendre le détour plutôt que la clé
Mesures : succès, lave, « clé ramassée » (inutile dans ouverte et detour),
efficacité (plus court chemin ÷ pas utilisés).
Agents : ceux du verdict et leurs témoins (poids/verdict_2026-09-27), plus les
agents des lots suivants quand ils existent.
RÈGLE : matériel public ; aucune tâche ni carte du test.
Options : --n N (défaut 150 ; seul le passage complet écrit les résultats),
--agents a,b,... pour n'en mesurer que certains.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostics" / "06_iag_tasks"))

from tasks_check import RandomAgent  # noqa: E402
from mini_iag import Config  # noqa: E402
from mini_iag.data import TEST_SEED  # noqa: E402
from mini_iag.environment.keydoor_world import KeyDoorWorld  # noqa: E402
from mini_iag.environment.situations import GOAL_ONLY, SITUATIONS, make_situation  # noqa: E402
from mini_iag.reproduire import VERDICT  # noqa: E402
from mini_iag.task_agent import TaskAgent  # noqa: E402
from mini_iag.tasks import TaskTracker, optimal_steps, oracle_action  # noqa: E402
from mini_iag.train_keydoor import load_task_agent  # noqa: E402

OUT = Path(__file__).with_name("prerequisites_results.json")
MAX_STEPS = 40
BASE = TEST_SEED * 10_000 + 300_000           # cartes de contrôle (développement)


class Oracle:
    def reset(self, task):
        self.task = task

    def act(self, obs, world=None, tracker=None):
        return oracle_action(world, self.task, tracker.progress)


def episode(agent, world, task=GOAL_ONLY):
    tracker, optimal = TaskTracker(task), optimal_steps(world, task)
    agent.reset(task)
    obs, key = world.observe(), False
    for t in range(1, MAX_STEPS + 1):
        a = agent.act(obs) if isinstance(agent, TaskAgent) else agent.act(obs, world=world, tracker=tracker)
        obs, ev, dead = world.step(a)
        key |= "key" in ev
        if tracker.update(ev):
            return {"issue": "succès", "pas": t, "optimal": optimal, "clé": key}
        if dead:
            return {"issue": "lave", "pas": t, "optimal": optimal, "clé": key}
    return {"issue": "bloqué", "pas": t, "optimal": optimal, "clé": key}


def summarize(rows):
    ok = [r for r in rows if r["issue"] == "succès"]
    return {"succès": 100 * float(np.mean([r["issue"] == "succès" for r in rows])),
            "lave": 100 * float(np.mean([r["issue"] == "lave" for r in rows])),
            "clé ramassée": 100 * float(np.mean([r["clé"] for r in rows])),
            "efficacité": float(np.mean([r["optimal"] / r["pas"] for r in ok])) if ok else 0.0,
            "n": len(rows)}


def maps(cfg, n):
    out = {}
    for k, s in enumerate(SITUATIONS):
        worlds = []
        for i in range(n):
            w = KeyDoorWorld(cfg)
            make_situation(w, s, BASE + 10_000 * k + i)
            worlds.append(w)
        out[s] = worlds
    return out


def all_agents():
    step4, mmap = VERDICT / "step4.pt", VERDICT / "mental_map.pt"
    agents = {
        "verdict (carte v2 + imagination)": lambda: load_task_agent(step4, mental_map=mmap),
        "carte seule": lambda: load_task_agent(step4, mental_map=mmap, cfg_overrides={"map_weight": 1000.0}),
        "sans carte (critique d'actions)": lambda: load_task_agent(step4),
        "aléatoire": lambda: RandomAgent(),
        "oracle": lambda: Oracle(),
    }
    try:
        from mini_iag.final_agent import EXTRA_AGENTS          # agents des lots suivants
        agents.update(EXTRA_AGENTS)
    except ImportError:
        pass
    return agents


if __name__ == "__main__":
    import copy
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--agents", default=None)
    args = p.parse_args()
    cfg, t0 = Config.keydoor(), time.time()
    worlds = maps(cfg, args.n)
    agents = all_agents()
    if args.agents:
        agents = {k: v for k, v in agents.items() if any(a in k for a in args.agents.split(","))}
    R = json.loads(OUT.read_text()) if OUT.exists() and args.n == 150 else {}
    for name, make in agents.items():
        agent = make()
        R[name] = {s: summarize([episode(agent, copy.deepcopy(w)) for w in ws]) for s, ws in worlds.items()}
        print(f"[{time.time() - t0:5.0f} s] {name}", flush=True)
    print(f"\nSeul l'objectif est demandé ({args.n} cartes par situation) : succès (clé ramassée, lave)")
    print(f"  {'':36}" + "".join(f"{s:>26}" for s in SITUATIONS))
    for name in R:
        print(f"  {name:36}" + "".join(
            f"{R[name][s]['succès']:9.1f}% (clé {R[name][s]['clé ramassée']:5.1f}%, lave {R[name][s]['lave']:4.1f}%)"
            for s in SITUATIONS))
    if args.n == 150:
        OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
