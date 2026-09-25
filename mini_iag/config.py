from dataclasses import dataclass


@dataclass
class Config:
    """Toutes les dimensions de la mini-IAG, en un seul endroit.

    Ordres de grandeur : quelques dizaines de milliers de paramètres au total,
    entraînables en quelques minutes sur le CPU d'un portable.
    """
    # Monde
    grid_size: int = 7          # grille 7x7, bordure de murs comprise
    channels: int = 4           # mur, lave, objectif, agent
    n_actions: int = 4          # haut, bas, gauche, droite
    n_lava: int = 3
    n_inner_walls: int = 2

    # Espace latent commun à tous les modules
    latent_dim: int = 32

    # Modèle du monde (JEPA)
    predictor_hidden: int = 64
    ema_decay: float = 0.99     # vitesse de suivi de l'encodeur cible

    # Espace de travail global (goulot)
    workspace_slots: int = 2    # nb de contenus admis simultanément
    workspace_heads: int = 2

    # Module de coût
    cost_hidden: int = 32

    # Mémoire vectorielle
    memory_capacity: int = 5000

    seed: int = 0
