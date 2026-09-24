"""Jauges homéostatiques : énergie, curiosité, attachement."""
try:
    import psutil
except ImportError:
    psutil = None


class Hormones:
    def __init__(self):
        self.energie = 1.0
        self.curiosite = 0.3
        self.attachement = 0.5

    def update(self, workspace):
        # Énergie liée à la batterie (ou lente décroissance si pas de batterie)
        batt = psutil.sensors_battery() if psutil else None
        if batt is not None:
            self.energie = batt.percent / 100
        else:
            self.energie = max(0.1, self.energie - 0.005)

        # Curiosité monte quand le workspace se répète (ennui)
        recent = [p.content for p in list(workspace.history)[-5:]]
        repetition = 1 - len(set(recent)) / max(1, len(recent))
        self.curiosite = min(1.0, max(0.0, self.curiosite + 0.1 * repetition - 0.03))

        # Attachement décroît sans interaction utilisateur
        self.attachement = max(0.0, self.attachement - 0.01)

    def __repr__(self):
        return (f"E={self.energie:.2f} C={self.curiosite:.2f} "
                f"A={self.attachement:.2f}")
