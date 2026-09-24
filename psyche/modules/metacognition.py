from .base import Module
from ..proposal import Proposal


class Metacognition(Module):
    """Observe l'historique du workspace (sauf ses propres sorties)."""
    name = "metacognition"

    def __init__(self, workspace):
        super().__init__()
        self.ws = workspace

    def propose(self, h):
        hist = list(self.ws.history)
        if len(hist) < 3:
            return None
        # Ne réfléchit PAS sur ses propres pensées (évite la récursion infinie)
        others = [p for p in hist[-6:] if p.source != self.name]
        if not others:
            return None
        sources = [p.source for p in others]
        dominant = max(set(sources), key=sources.count)
        part = sources.count(dominant) / len(sources)
        if part > 0.5:
            return Proposal(self.name, f"mon attention est dominée par '{dominant}'",
                            0.4 + 0.4 * part)
        return Proposal(self.name, "mes pensées sont variées", 0.25)
