"""Global Workspace : goulot attentionnel avec inhibition de retour."""
import random
from collections import deque


class GlobalWorkspace:
    def __init__(self, inhibition=0.5, memory=4):
        self.history = deque(maxlen=50)   # archive lue par la métacognition
        self.inhibition = inhibition      # pénalité "inhibition de retour"
        self.memory = memory              # nb de cycles où la pénalité s'applique

    def _penalty(self, p):
        pen = 0.0
        for r in list(self.history)[-self.memory:]:
            if r.source == p.source:
                pen += self.inhibition / 2
            if r.content == p.content:
                pen += self.inhibition
        return pen

    def compete(self, proposals):
        """Nouvelle compétition à CHAQUE cycle : un seul gagnant."""
        scored = [(p.salience - self._penalty(p) + random.gauss(0, 0.05), p)
                  for p in proposals if p is not None]
        if not scored:
            return None
        _, winner = max(scored, key=lambda x: x[0])
        self.history.append(winner)
        return winner
