from .base import Module
from ..proposal import Proposal


class UserInput(Module):
    """Canal utilisateur : un message reçu est très saillant (une seule fois)."""
    name = "utilisateur"

    def __init__(self):
        super().__init__()
        self.pending = None

    def hear(self, message):
        self.pending = message

    def propose(self, h):
        if self.pending is None:
            return None
        msg, self.pending = self.pending, None
        return Proposal(self.name, f"l'utilisateur dit : {msg}", 0.95)
