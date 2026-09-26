"""Architecture du monde v2 (clé, porte) : mêmes briques, coût configurable."""
from .architecture import Architecture
from .modules import ConfigurableCost, Critic


class ArchitectureV2(Architecture):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cost = ConfigurableCost(cfg, w_danger=cfg.danger_weight, w_success=cfg.success_weight)
        self.critic = Critic(cfg)
