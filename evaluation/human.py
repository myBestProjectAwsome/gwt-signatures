"""L'humain expert passe le test.  Lancer :  python -m evaluation.human [--cas H1]

Tu joues les tâches du test, sans entraînement préalable (comme l'agent en
"zéro essai"), sur les 12 premières cartes de chaque cas. Tes résultats
servent de référence pour la règle V1 (agilité humaine).

Commandes : z ou w = haut, s = bas, q ou a = gauche, d = droite, puis Entrée.
Tu peux taper plusieurs coups d'affilée (ex. "zzd").  x = abandonner la carte.
On ne tape jamais k ou * : on ramasse la clé ou on atteint l'objectif en marchant dessus.

Échauffement (recommandé avant le test) :  python -m evaluation.human --echauffement
  quelques cartes des tâches PUBLIQUES (objectif, clé) sur des cartes publiques,
  pour apprendre les commandes. Rien n'est enregistré. Le test mesure l'agilité
  face à une tâche nouvelle, pas la découverte du clavier : l'agent, lui, n'a pas
  de clavier à apprendre.
"""
import argparse
import json
from pathlib import Path

from types import SimpleNamespace

from mini_iag.config import Config
from mini_iag.environment.keydoor_world import KeyDoorWorld
from mini_iag.tasks import TRAINING_TASKS, TaskTracker, optimal_steps

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


COMMANDS = ("Commandes : z = haut, s = bas, q = gauche, d = droite (puis Entrée), plusieurs coups "
            "d'affilée possibles (ex. qqs). On marche SUR la clé ou l'objectif, on ne tape pas k ni *.")
WARMUP_SEED = 500_000            # cartes publiques d'échauffement (jamais celles du test)


def warmup(cfg, n=3):
    """Échauffement : tâches et cartes publiques, rien n'est enregistré."""
    print("\nÉCHAUFFEMENT (tâches publiques, rien n'est enregistré)\n" + COMMANDS)
    for task in TRAINING_TASKS:
        case = SimpleNamespace(code="échauffement", task=task, max_steps=40)
        for i in range(n):
            world = KeyDoorWorld(cfg)
            for attempt in range(50):
                world.reset(seed=WARMUP_SEED + 100 * TRAINING_TASKS.index(task) + i * 7 + attempt)
                if optimal_steps(world, task):
                    break
            play(world, case, i + 1, n)
    print("\nÉchauffement terminé. Quand tu es prêt : python -m evaluation.human")


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluation.human")
    parser.add_argument("--cas", help="ne jouer qu'un cas (ex. H1)")
    parser.add_argument("--echauffement", action="store_true", help="s'entraîner aux commandes")
    args = parser.parse_args()
    cfg = Config()
    if args.echauffement:
        warmup(cfg)
        return
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    cases = [c for c in all_cases() if c.human and (args.cas in (None, c.code))]
    print("Légende : # mur  ~ lave  * objectif  k clé  D porte  A toi  (a = toi avec la clé)")
    print(COMMANDS)
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
