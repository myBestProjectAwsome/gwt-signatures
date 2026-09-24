import random

from .base import Module
from ..proposal import Proposal


class Explorer(Module):
    """Pousse à explorer quand curiosité et énergie sont hautes."""
    name = "exploration"
    ideas = ["observer l'heure", "examiner la mémoire vive",
             "se demander ce que fait l'utilisateur", "revoir un souvenir"]

    def propose(self, h):
        return Proposal(self.name, random.choice(self.ideas), h.curiosite * h.energie)
