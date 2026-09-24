class Module:
    """Classe de base : chaque module propose un contenu et reçoit le broadcast."""
    name = "module"

    def __init__(self):
        self.last_broadcast = None

    def propose(self, h):
        raise NotImplementedError

    def receive(self, broadcast, h):
        """Réentrance : chaque module voit le contenu conscient.
        `h` permet à un module de satisfaire une pulsion quand il gagne."""
        self.last_broadcast = broadcast
