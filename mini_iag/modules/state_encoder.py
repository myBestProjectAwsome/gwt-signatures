from torch import nn


class StateEncoder(nn.Module):
    """Observation (4, 7, 7) -> vecteur latent de dimension `latent_dim`.

    Deux convolutions 3x3 (qui voient le voisinage de chaque case), puis
    une couche linéaire qui résume toute la grille en un seul vecteur.
    """

    def __init__(self, cfg):
        super().__init__()
        n = cfg.grid_size
        self.net = nn.Sequential(
            nn.Conv2d(cfg.channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * n * n, cfg.latent_dim),
        )

    def forward(self, obs):
        return self.net(obs)
