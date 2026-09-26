from dataclasses import dataclass


@dataclass
class Config:
    """Toutes les dimensions de la mini-IAG, en un seul endroit.

    Ordres de grandeur : quelques dizaines de milliers de paramètres au total,
    entraînables en quelques minutes sur le CPU d'un portable.
    """
    # Monde
    grid_size: int = 7          # grille 7x7, bordure de murs comprise
    channels: int = 4           # mur, lave, objectif, agent (monde v2 : 7, cf. Config.keydoor)
    n_events: int = 2           # événements prédits (v1 : lave, objectif ; v2 : 4)
    pair_events: bool = False   # v2 : événements jugés sur une TRANSITION (avant, après)
    n_actions: int = 4          # haut, bas, gauche, droite
    n_lava: int = 3
    n_inner_walls: int = 2

    # Espace latent commun à tous les modules
    latent_dim: int = 32

    # Modèle du monde (JEPA)
    predictor_hidden: int = 64
    ema_decay: float = 0.99     # vitesse de suivi de l'encodeur cible
    multistep_horizon: int = 6  # nb de pas imaginés pendant l'entraînement (étape 3 : 3 -> 6)
    success_event_weight: float = 10.0  # le succès est rare : poids x10 dans la tête d'événements

    # Espace de travail global (goulot)
    workspace_slots: int = 2    # nb de contenus admis simultanément
    workspace_heads: int = 2

    # Module de coût
    cost_hidden: int = 32
    danger_weight: float = 4.0   # "valeurs" de l'agent : mourir compte 4x plus que réussir
    success_weight: float = 1.0  # (compromis mesuré à l'étape 3, cf. README)

    # Mémoire vectorielle
    memory_capacity: int = 5000

    # Planification (étape 3)
    planning_horizon: int = 5   # suites de 5 actions : 4^5 = 1024 plans imaginés à chaque pas
    discount: float = 0.95
    step_penalty: float = 0.02  # chaque pas coûte un peu : préférer les chemins courts
    novelty_weight: float = 0.02  # pénalité pour retourner dans un état déjà visité
    max_steps: int = 30
    value_weight: float = 1.0   # poids de la critique au bout de l'horizon (monde v2)
    value_discount: float = 0.9

    seed: int = 0

    @classmethod
    def keydoor(cls, **kw):
        """Configuration du monde v2 (clé, porte) : 7 canaux, 4 événements jugés sur
        des transitions, latent de 64 (32 ne suffisait plus : clé et lave mal
        représentées, cf. README étape 4)."""
        base = dict(channels=7, n_events=4, pair_events=True, latent_dim=64,
                    predictor_hidden=128, cost_hidden=64,
                    value_weight=0.0)   # critique entraînée mais non utilisée : son
                                        # ablation montre qu'elle n'aide pas (README, étape 4a)
        return cls(**{**base, **kw})
