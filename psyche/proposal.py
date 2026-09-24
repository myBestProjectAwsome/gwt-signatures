from dataclasses import dataclass


@dataclass
class Proposal:
    """Contenu proposé par un module pour accéder au workspace."""
    source: str
    content: str
    salience: float  # 0..1
