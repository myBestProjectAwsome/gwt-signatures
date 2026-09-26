"""Contrôle d'isolement : le code de l'agent (mini_iag/) ne doit rien savoir du test.

Échec si un fichier de mini_iag/ :
  - importe le paquet `evaluation` ;
  - contient une graine réservée au test (7 000 000 ou 8 000 000 et plus) ;
  - nomme une famille de cartes du test ("enclos", "pieces", "lave_dense") ;
  - contient une séquence d'événements identique à une tâche du test.
Lancer :  python -m evaluation.isolation
"""
import ast
import sys
from pathlib import Path

from .layouts import LAYOUTS, PRACTICE_BASE, TEST_BASE
from .tasks import all_cases

ROOT = Path(__file__).resolve().parents[1]


def check(package="mini_iag"):
    secret_layouts = set(LAYOUTS) - {"standard"}
    secret_sequences = {c.task.sequence for c in all_cases() if c.held_out_task}
    problems = []
    for path in sorted((ROOT / package).rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(ROOT)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                if any(n.split(".")[0] == "evaluation" for n in names):
                    problems.append(f"{rel}:{node.lineno} importe evaluation")
            elif isinstance(node, ast.Constant):
                v = node.value
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= TEST_BASE \
                        and v < PRACTICE_BASE + 10_000_000:
                    problems.append(f"{rel}:{node.lineno} graine réservée au test ({v})")
                if isinstance(v, str) and v in secret_layouts:
                    problems.append(f"{rel}:{node.lineno} famille de cartes du test ({v!r})")
            elif isinstance(node, ast.Tuple) and node.elts and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
                if tuple(e.value for e in node.elts) in secret_sequences:
                    problems.append(f"{rel}:{node.lineno} séquence d'une tâche du test")
    return problems


if __name__ == "__main__":
    probs = check()
    if probs:
        print("ISOLEMENT ROMPU :")
        for p in probs:
            print("  " + p)
        sys.exit(1)
    print("Isolement respecté : mini_iag/ ne connaît rien du test.")
