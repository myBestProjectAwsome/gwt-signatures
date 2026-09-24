"""Jauges homéostatiques : énergie, curiosité, attachement.

Modèle de pulsion (drive) : une jauge monte avec le temps et n'est
satisfaite que par une action précise (consommation).
  - curiosité  : monte en continu, plus vite si les pensées se répètent
                 (ennui) ; satisfaite quand l'exploration gagne le workspace
  - attachement: baisse en continu ; ne remonte QUE si l'utilisateur écrit
  - énergie    : suit la batterie (1.0 sur secteur / sans batterie)
"""


class Hormones:
    CURIOSITY_BASE = 0.02     # montée de base par cycle
    CURIOSITY_BOREDOM = 0.04  # montée supplémentaire si ennui maximal
    CURIOSITY_SATIETY = 0.25  # baisse quand l'exploration gagne
    ATTACH_DECAY = 0.005
    ATTACH_USER = 0.4

    def __init__(self):
        self.energie = 1.0
        self.curiosite = 0.3
        self.attachement = 0.5

    @staticmethod
    def _clamp(v):
        return min(1.0, max(0.0, v))

    def ennui(self, workspace, window=6):
        """Manque de nouveauté : part de contenus répétés dans les derniers gagnants."""
        recent = [p.content for p in list(workspace.history)[-window:]]
        if not recent:
            return 0.0
        return 1 - len(set(recent)) / len(recent)

    def update(self, workspace, env):
        batt = env.battery()
        self.energie = 1.0 if batt is None else batt
        self.curiosite = self._clamp(self.curiosite + self.CURIOSITY_BASE
                                     + self.CURIOSITY_BOREDOM * self.ennui(workspace))
        self.attachement = self._clamp(self.attachement - self.ATTACH_DECAY)

    def satisfy_curiosity(self):
        self.curiosite = self._clamp(self.curiosite - self.CURIOSITY_SATIETY)

    def on_user_message(self):
        self.attachement = self._clamp(self.attachement + self.ATTACH_USER)

    def __repr__(self):
        return (f"E={self.energie:.2f} C={self.curiosite:.2f} "
                f"A={self.attachement:.2f}")
