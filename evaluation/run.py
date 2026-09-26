"""Lancer le test.  python -m evaluation.run --references [--n 50]

--references : agents de référence (aléatoire, oracle) sur tous les cas.
               Sert à vérifier que la batterie fonctionne et que chaque carte est soluble.
Le passage de l'agent de la mini-IAG (étape 5) sera ajouté ici, SANS modifier
les règles figées dans battery.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mini_iag.config import Config

from .agents import OracleAgent, RandomAgent
from .battery import N_MAPS, run_case, summarize
from .tasks import all_cases

HERE = Path(__file__).parent
ROOT = HERE.parent


def integrity():
    """Compare les fichiers figés à l'empreinte enregistrée."""
    reg = HERE / "registration.json"
    if not reg.exists():
        return None
    data = json.loads(reg.read_text())
    ok = all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h
             for f, h in data["fichiers_figés"].items())
    return ok, data


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluation.run")
    parser.add_argument("--references", action="store_true")
    parser.add_argument("--n", type=int, default=N_MAPS)
    args = parser.parse_args()
    state = integrity()
    if state is None:
        print("Protocole : pas encore enregistré (python -m evaluation.register).")
    else:
        print(f"Protocole enregistré le {state[1]['date']} : "
              + ("INTACT" if state[0] else "MODIFIÉ DEPUIS L'ENREGISTREMENT !"))
    if not args.references:
        parser.print_help()
        return
    cfg = Config()
    print(f"\n{'cas':5}{'tâche':22}{'cartes':12}" + "".join(f"{n:>22}" for n in ("aléatoire", "oracle")))
    for case in all_cases():
        row = f"{case.code:5}{case.task.name:22}{case.layout:12}"
        for agent in (RandomAgent(), OracleAgent()):
            s = summarize(run_case(agent, cfg, case, n=args.n))
            row += f"{s['succès']:9.1f} % lave {s['lave']:4.1f} %"
        print(row)


if __name__ == "__main__":
    main()
