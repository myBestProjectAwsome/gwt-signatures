"""Référence reproductible (feuille de route v2, lot 1).   python -m mini_iag.reproduire

Les poids exacts qui ont passé le test du 27 septembre 2026 sont dans le dépôt :
poids/verdict_2026-09-27/ (step4.pt et mental_map.pt, 1,3 Mo en tout), avec
leurs empreintes dans EMPREINTES.json.

  python -m mini_iag.reproduire              vérifie les empreintes des poids du verdict
  python -m mini_iag.reproduire --installer  les copie dans checkpoints/ (écrase les tiens)
  python -m mini_iag.reproduire --installer-dev   copie les poids de développement (lots 2-3)
  python -m mini_iag.reproduire --reentrainer
        réentraîne tout (~30 min) dans poids/reentraines/ et compare aux poids du verdict.
        Le réentraînement est déterministe sur une même machine (graine fixée), mais
        d'une machine à l'autre les calculs flottants peuvent différer : des poids
        différents ne sont donc pas une erreur. On compare alors les comportements
        (python diagnostics/13_iag_dry_run/dry_run_check.py).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

VERDICT = Path("poids/verdict_2026-09-27")
RETRAINED = Path("poids/reentraines")
FILES = {"step4.pt": "architecture", "mental_map.pt": "map"}


def tensor_hash(state):
    """Empreinte des VALEURS des poids (indépendante de la façon dont le fichier est écrit)."""
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def weights_hash(path, key):
    return tensor_hash(torch.load(path, weights_only=False)[key])


def check(folder=VERDICT):
    ref = json.loads((VERDICT / "EMPREINTES.json").read_text())
    ok = True
    for name, key in FILES.items():
        h = weights_hash(folder / name, key)
        same = h == ref["poids"][name]
        ok &= same
        print(f"  {name:15} {'identique' if same else 'DIFFÉRENT'}   {h[:16]}…")
    return ok


def main():
    p = argparse.ArgumentParser(prog="python -m mini_iag.reproduire")
    p.add_argument("--installer", action="store_true")
    p.add_argument("--reentrainer", action="store_true")
    p.add_argument("--installer-dev", action="store_true",
                   help="copie aussi les poids de développement (carte v3, modèle des conséquences)")
    args = p.parse_args()
    print(f"Poids du verdict ({VERDICT}) :")
    if not check():
        print("Les poids du verdict ont été modifiés !")
        return
    if args.installer:
        Path("checkpoints").mkdir(exist_ok=True)
        for name in FILES:
            shutil.copy(VERDICT / name, Path("checkpoints") / name)
        print("Copiés dans checkpoints/ : l'agent est exactement celui du verdict.")
    if args.installer_dev:
        Path("checkpoints").mkdir(exist_ok=True)
        for f in Path("poids/developpement").glob("*.pt"):
            shutil.copy(f, Path("checkpoints") / f.name)
        print("Poids de développement copiés dans checkpoints/ (carte v3, modèle des conséquences).")
    if args.reentrainer:
        from .train_keydoor import train
        from .train_map import train_map
        RETRAINED.mkdir(parents=True, exist_ok=True)
        train(out=RETRAINED / "step4.pt")
        train_map(out=RETRAINED / "mental_map.pt")
        print(f"\nPoids réentraînés ({RETRAINED}) comparés au verdict :")
        if check(RETRAINED):
            print("Reproduction exacte.")
        else:
            print("Poids différents (normal d'une machine à l'autre) : comparer les comportements "
                  "avec diagnostics/13_iag_dry_run.")


if __name__ == "__main__":
    main()
