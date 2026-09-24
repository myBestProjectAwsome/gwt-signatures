from .base import Module
from ..proposal import Proposal


class SystemSensor(Module):
    """Perçoit la charge CPU (via l'environnement) : une charge élevée est saillante."""
    name = "capteur_systeme"

    def __init__(self, env):
        super().__init__()
        self.env = env

    def propose(self, h):
        cpu = self.env.cpu()
        return Proposal(self.name, f"CPU à {100 * cpu:.0f}%", 0.2 + 0.7 * cpu)
