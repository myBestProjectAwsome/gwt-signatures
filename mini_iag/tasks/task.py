"""Format des tâches : une tâche = une SÉQUENCE d'événements à provoquer, dans l'ordre.

Exemples (format public) :
    Task("objectif", ("goal",))          atteindre l'objectif
    Task("clé",      ("key",))           ramasser la clé
Une tâche est réussie quand tous ses événements ont eu lieu dans l'ordre ;
elle échoue si l'agent meurt (lave) ou dépasse le nombre de pas autorisés.

C'est ce que le "configurateur" (LeCun 2022) transmettra au module de coût :
la tâche change CE QUI COMPTE, pas la physique du monde.

Les tâches d'entraînement sont listées ici. Les tâches du test final sont
dans evaluation/ et ne doivent JAMAIS être importées par le code de
l'agent (vérifié par evaluation/isolation.py).
"""
from dataclasses import dataclass

VALID_EVENTS = ("goal", "key", "door")


@dataclass(frozen=True)
class Task:
    name: str
    sequence: tuple            # événements à provoquer, dans l'ordre
    instruction: str = ""      # consigne en français (pour l'humain)

    def __post_init__(self):
        if not self.sequence or any(e not in VALID_EVENTS for e in self.sequence):
            raise ValueError(f"tâche invalide : {self.sequence}")


TRAINING_TASKS = (
    Task("objectif", ("goal",), "Atteins l'objectif (*) sans tomber dans la lave."),
    Task("clé", ("key",), "Ramasse la clé (k) sans tomber dans la lave."),
)
