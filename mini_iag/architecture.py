"""Assemblage des 4 modules fondamentaux (étape 1 : structures vides).

Aucune connexion fonctionnelle entre eux pour l'instant : le câblage
(le workspace qui interroge le modèle du monde et lit le coût) est
l'objet de l'étape 3.
"""
import torch
from torch import nn

from .modules import WorldModel, BottleneckWorkspace, CostModule, VectorMemory


class Architecture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.world_model = WorldModel(cfg)
        self.workspace = BottleneckWorkspace(cfg)
        self.cost = CostModule(cfg, w_danger=cfg.danger_weight, w_success=cfg.success_weight)
        self.memory = VectorMemory(cfg)          # pas un nn.Module : pas de poids

    def parameter_counts(self):
        def count(m, trainable_only=True):
            return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
        return {
            "modele du monde (entraînable)": count(self.world_model),
            "  dont encodeur cible (EMA)": count(self.world_model.target_encoder, False),
            "espace de travail": count(self.workspace),
            "module de coût": count(self.cost),
            "mémoire (capacité, en vecteurs)": self.memory.capacity,
        }
