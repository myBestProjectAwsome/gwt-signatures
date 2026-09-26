import numpy as np


class RandomAgent:
    """Référence basse : une action au hasard à chaque pas."""
    name = "aléatoire"
    privileged = False

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, task):
        pass

    def act(self, obs):
        return int(self.rng.integers(4))
