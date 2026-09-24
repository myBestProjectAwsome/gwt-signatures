from .base import Module
from ..proposal import Proposal


class Social(Module):
    """Devient saillant quand l'attachement baisse."""
    name = "social"

    def propose(self, h):
        return Proposal(self.name, "l'utilisateur ne m'a pas parlé",
                        0.6 * (1 - h.attachement))
