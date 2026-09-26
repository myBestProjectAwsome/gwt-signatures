"""L'humain expert passe le test.  Lancer :  python -m evaluation.human [--cas H1]

Tu joues les tâches du test, sans entraînement préalable (comme l'agent en
"zéro essai"), sur les 12 premières cartes de chaque cas. Tes résultats
servent de référence pour la règle V1 (agilité humaine).

Commandes : z ou w = haut, s = bas, q ou a = gauche, d = droite, puis Entrée.
Tu peux taper plusieurs coups d'affilée (ex. "zzd").  x = abandonner la carte.
"""
import argparse
import json
from pathlib import Path

from mini_iag.config import Config
from mini_iag.tasks import TaskTracker, optimal_steps

from .battery import N_HUMAN, case_index
from .layouts import make_map
from .tasks import all_cases

KEYS = {"z": 0, "w": 0, "s": 1, "q": 2, "a": 2, "d": 3}
RESULTS = Path(__file__).with_name("human_results.json")


def play(world, case, n, total):
    tracker, optimal = TaskTracker(case.task), optimal_steps(world, case.task)
    steps = 0
    while steps < case.max_steps:
        print(f"\n[{case.code} — carte {n}/{total}]  {case.task.instruction}")
        print(f"pas utilisés : {steps}/{case.max_steps}\n")
        print(world.render())
        cmd = input("\ncoups > ").strip().lower()
        if cmd == "x":
            break
        for ch in cmd:
            if ch not in KEYS:
                continue
            _, events, dead = world.step(KEYS[ch])
            steps += 1
            if tracker.update(events):
                print(world.render(), f"\n✔ Réussi en {steps} pas (meilleur possible : {optimal}).")
                return {"success": True, "steps": steps, "optimal": optimal, "lava": False}
            if dead:
                print(world.render(), "\n✘ Lave.")
                return {"success": False, "steps": steps, "optimal": optimal, "lava": True}
            if steps >= case.max_steps:
                break
    print("✘ Pas de réussite.")
    return {"success": False, "steps": max(steps, 1), "optimal": optimal, "lava": False}


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluation.human")
    parser.add_argument("--cas", help="ne jouer qu'un cas (ex. H1)")
    args = parser.parse_args()
    cfg = Config()
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    cases = [c for c in all_cases() if c.human and (args.cas in (None, c.code))]
    print("Légende : # mur  ~ lave  * objectif  k clé  D porte  A toi  (a = toi avec la clé)")
    for case in cases:
        k = case_index(case)
        res = [play(make_map(cfg, case, i, case_index=k), case, i + 1, N_HUMAN)
               for i in range(N_HUMAN)]
        results[case.code] = res
        RESULTS.write_text(json.dumps(results, indent=1))
        print(f"\n{case.code} : {sum(r['success'] for r in res)}/{N_HUMAN} réussies. "
              f"Enregistré dans {RESULTS.name}.")


if __name__ == "__main__":
    main()
