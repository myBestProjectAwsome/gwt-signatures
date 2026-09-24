from collections import Counter

from .base import Module
from ..proposal import Proposal


class Metacognition(Module):
    """Observe l'historique du workspace (sauf ses propres sorties).

    - Saillance modérée (0.1 à 0.5) : l'observation de soi ne doit l'emporter
      que si rien de plus important n'est en jeu (cf. diagnostics/01_inhibition :
      à 0.4-0.8 elle monopolisait le workspace).
    - Habituation : une observation déjà diffusée perd 70 % de sa saillance
      tant qu'elle reste identique. Sans ça, "mon attention est dominée par X"
      se répète indéfiniment, puisque le fait de le dire ne change pas X.
    """
    name = "metacognition"
    HABITUATION = 0.3

    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace
        self.last_report = None

    def propose(self, h):
        # les 6 derniers contenus des AUTRES modules (évite la récursion infinie)
        others = [p for p in self.ws.history if p.source != self.name][-6:]
        if len(others) < 3:
            return None
        # Counter.most_common : départage déterministe (ordre d'apparition)
        dominant, count = Counter(p.source for p in others).most_common(1)[0]
        part = count / len(others)
        if part > 0.5:
            content, sal = f"mon attention est dominée par '{dominant}'", 0.15 + 0.35 * part
        else:
            content, sal = "mes pensées sont variées", 0.1
        if content == self.last_report:
            sal *= self.HABITUATION
        return Proposal(self.name, content, sal)

    def receive(self, broadcast, h):
        super().receive(broadcast, h)
        if broadcast.source == self.name:
            self.last_report = broadcast.content
