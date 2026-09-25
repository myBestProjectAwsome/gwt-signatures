"""mini_iag — prototype d'architecture cognitive modulaire, à l'échelle d'un portable.

Inspiré de LeCun (2022), "A Path Towards Autonomous Machine Intelligence"
(modèle du monde JEPA, module de coût, mémoire) et de Goyal et al. (2022),
"Coordination Among Neural Modules Through a Shared Global Workspace".

Ce n'est PAS une IAG : c'est une maquette réduite pour tester, étape par
étape, si ces briques se composent et ce que chacune apporte (ablations).
"""
from .config import Config
from .architecture import Architecture
from .environment import GridWorld

__all__ = ["Config", "Architecture", "GridWorld"]
