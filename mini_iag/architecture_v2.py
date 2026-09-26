"""Architecture du monde v2 (clé, porte) : mêmes briques, coût configurable,
et une critique d'actions apprise sur l'expérience réelle (planification longue)."""
from .architecture import Architecture
from .modules import ActionCritic, ConfigurableCost


class ArchitectureV2(Architecture):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cost = ConfigurableCost(cfg, w_danger=cfg.danger_weight, w_success=cfg.success_weight)
        self.critic = ActionCritic(cfg)
