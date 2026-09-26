"""Enregistre (fige) le protocole.  python -m evaluation.register

Calcule l'empreinte SHA-256 des fichiers du test et l'écrit dans
registration.json, avec la date. Refuse d'écraser un enregistrement existant :
modifier le test après coup annulerait le pré-enregistrement.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from .battery import HUMAN_THRESHOLD

HERE = Path(__file__).parent
ROOT = HERE.parent
FROZEN = ["evaluation/PROTOCOLE.md", "evaluation/tasks.py", "evaluation/layouts.py",
          "evaluation/battery.py", "evaluation/isolation.py",
          "mini_iag/tasks/task.py", "mini_iag/tasks/tracker.py", "mini_iag/tasks/search.py"]
INFORMATIVE = ["mini_iag/environment/keydoor_world.py"]   # physique : changements à documenter


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluation.register")
    parser.add_argument("--force", action="store_true", help="écraser (annule le pré-enregistrement)")
    args = parser.parse_args()
    out = HERE / "registration.json"
    if out.exists() and not args.force:
        sys.exit("Déjà enregistré. Modifier le test maintenant annulerait le pré-enregistrement.")
    if HUMAN_THRESHOLD is None:
        sys.exit("Le seuil V1 (HUMAN_THRESHOLD) doit être fixé avant l'enregistrement.")
    data = {"date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "seuil_V1": HUMAN_THRESHOLD,
            "fichiers_figés": {f: sha(f) for f in FROZEN},
            "fichiers_informatifs": {f: sha(f) for f in INFORMATIVE}}
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"Protocole enregistré le {data['date']} (seuil V1 = {HUMAN_THRESHOLD}).")
    for f, h in data["fichiers_figés"].items():
        print(f"  {h[:16]}…  {f}")


if __name__ == "__main__":
    main()
