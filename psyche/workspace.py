"""Global Workspace : goulot attentionnel avec inhibition de retour.

L'inhibition de retour pénalise un module (et un contenu) qui vient de gagner.
Elle DÉCROÎT avec le temps : forte juste après la victoire, divisée par 2
à chaque cycle. Elle sert à départager des propositions proches, pas à
imposer une rotation (cf. diagnostics/01_inhibition).
"""
import random
from collections import deque


class GlobalWorkspace:
    def __init__(self, inhibition=0.15, half_life=1.0, window=8, noise=0.05):
        self.history = deque(maxlen=50)   # archive lue par la métacognition
        self._wins = deque(maxlen=window) # (cycle, source, contenu)
        self.inhibition = inhibition
        self.half_life = half_life        # en cycles
        self.noise = noise
        self.t = 0

    def _penalty(self, p):
        pen = 0.0
        for t_win, source, content in self._wins:
            decay = 0.5 ** ((self.t - t_win - 1) / self.half_life)
            if source == p.source:
                pen += self.inhibition * decay
            if content == p.content:
                pen += self.inhibition * decay
        return pen

    def compete(self, proposals):
        """Nouvelle compétition à CHAQUE cycle : un seul gagnant."""
        scored = [(p.salience - self._penalty(p) + random.gauss(0, self.noise), p)
                  for p in proposals if p is not None]
        winner = max(scored, key=lambda x: x[0])[1] if scored else None
        if winner is not None:
            self.history.append(winner)
            self._wins.append((self.t, winner.source, winner.content))
        self.t += 1
        return winner
