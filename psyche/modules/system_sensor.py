from .base import Module
from ..proposal import Proposal

try:
    import psutil
except ImportError:
    psutil = None


class SystemSensor(Module):
    """Perçoit la charge CPU : une charge élevée est saillante."""
    name = "capteur_systeme"

    def propose(self, h):
        if not psutil:
            return Proposal(self.name, "capteurs indisponibles", 0.1)
        cpu = psutil.cpu_percent(interval=None)
        return Proposal(self.name, f"CPU à {cpu:.0f}%", 0.2 + 0.7 * cpu / 100)
