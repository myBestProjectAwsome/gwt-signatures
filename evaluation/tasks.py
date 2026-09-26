"""Tâches et cas du test final. SECRET pour l'entraînement.

Chaque cas associe une tâche, une famille de cartes et un budget de pas.
Toutes les tâches sont des SÉQUENCES d'événements du monde public
(clé, porte, objectif), mais aucune n'est demandée pendant l'entraînement :
  H1  ramasser la clé PUIS atteindre l'objectif : enchaîner deux tâches connues
  H2  ouvrir la porte : un événement que l'agent a pu observer (physique du
      monde) mais qu'on ne lui a jamais demandé de provoquer
  H3  objectif enfermé derrière une porte : clé, porte, objectif, sur des
      cartes où le chemin dépasse l'horizon d'imagination de l'étape 3
  W1  atteindre l'objectif sur des cartes à pièces et couloirs (jamais vues)
  W2  atteindre l'objectif sur des cartes à lave dense (jamais vues)
  R0  atteindre l'objectif sur des cartes standard (référence pour W1, W2, V5)
Des tâches supplémentaires peuvent être ajoutées par l'humain dans
evaluation/secret_tasks.json (connues de lui seul jusqu'au jour du test).
"""
import json
from dataclasses import dataclass
from pathlib import Path

from mini_iag.tasks import TRAINING_TASKS, Task

GOAL_TASK = TRAINING_TASKS[0]


@dataclass(frozen=True)
class Case:
    code: str
    task: Task
    layout: str          # famille de cartes (voir evaluation/layouts.py)
    max_steps: int
    held_out_task: bool  # tâche jamais demandée pendant l'entraînement ?
    human: bool          # l'humain joue-t-il ce cas ?


CASES = (
    Case("H1", Task("clé puis objectif", ("key", "goal"),
                    "Ramasse la clé (k), PUIS va sur l'objectif (*). Évite la lave (~)."),
         "standard", 40, True, True),
    Case("H2", Task("ouvrir la porte", ("key", "door"),
                    "Ouvre la porte (D) : il faut d'abord la clé (k). Évite la lave (~)."),
         "standard", 40, True, True),
    Case("H3", Task("objectif enfermé", ("key", "door", "goal"),
                    "L'objectif (*) est derrière une porte (D). Prends la clé (k), "
                    "ouvre la porte, atteins l'objectif. Évite la lave (~)."),
         "enclos", 50, True, True),
    Case("W1", GOAL_TASK, "pieces", 40, False, False),
    Case("W2", GOAL_TASK, "lave_dense", 40, False, False),
    Case("R0", GOAL_TASK, "standard", 40, False, False),
)

SECRET_FILE = Path(__file__).with_name("secret_tasks.json")


def secret_cases():
    """Tâches secrètes de l'humain : [{"code", "nom", "sequence", "consigne", "cartes"}]."""
    if not SECRET_FILE.exists():
        return ()
    out = []
    for i, d in enumerate(json.loads(SECRET_FILE.read_text())):
        out.append(Case(d.get("code", f"S{i + 1}"),
                        Task(d["nom"], tuple(d["sequence"]), d.get("consigne", "")),
                        d.get("cartes", "standard"), int(d.get("max_pas", 50)), True, True))
    return tuple(out)


def all_cases():
    return CASES + secret_cases()
