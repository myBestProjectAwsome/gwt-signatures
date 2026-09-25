"""Monde en grille : le "corps" virtuel de la mini-IAG.

Une grille 7x7 entourée de murs, avec quelques murs intérieurs, des cases de
lave (danger) et un objectif. L'agent se déplace d'une case par action.

Observation : tableau (4, 7, 7) de 0/1, un canal par type d'objet
    0 = mur, 1 = lave, 2 = objectif, 3 = agent

Chaque carte est tirée au hasard à partir d'une graine : on peut donc
générer des cartes d'entraînement et des cartes JAMAIS VUES pour tester
la généralisation (étape 5).
"""
import numpy as np

WALL, LAVA, GOAL, AGENT = range(4)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # haut, bas, gauche, droite
ACTION_NAMES = ["haut", "bas", "gauche", "droite"]


class GridWorld:
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.n = cfg.grid_size
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------ carte
    def reset(self, seed=None):
        """Nouvelle carte aléatoire, toujours soluble. Renvoie l'observation initiale."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        while True:
            self._generate()
            if self.distance_to_goal() is not None:
                break
        self.done = False
        return self.observe()

    def _generate(self):
        n = self.n
        self.grid = np.zeros((3, n, n), dtype=np.float32)       # mur, lave, objectif
        self.grid[WALL, 0, :] = self.grid[WALL, -1, :] = 1
        self.grid[WALL, :, 0] = self.grid[WALL, :, -1] = 1
        free = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
        idx = self.rng.permutation(len(free))
        cells = [free[i] for i in idx]
        k = 0
        for _ in range(self.cfg.n_inner_walls):
            self.grid[WALL][cells[k]] = 1; k += 1
        for _ in range(self.cfg.n_lava):
            self.grid[LAVA][cells[k]] = 1; k += 1
        self.goal = cells[k]; k += 1
        self.grid[GOAL][self.goal] = 1
        self.agent = cells[k]

    def distance_to_goal(self, start=None):
        """Plus court chemin (en cases) sans mur ni lave, ou None si impossible."""
        start = start or self.agent
        frontier, seen = [start], {start: 0}
        while frontier:
            cur = frontier.pop(0)
            if cur == self.goal:
                return seen[cur]
            for dr, dc in MOVES:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in seen and not self.grid[WALL][nxt] and not self.grid[LAVA][nxt]:
                    seen[nxt] = seen[cur] + 1
                    frontier.append(nxt)
        return None

    def observe(self):
        obs = np.zeros((4, self.n, self.n), dtype=np.float32)
        obs[:3] = self.grid
        obs[AGENT][self.agent] = 1
        return obs

    # ------------------------------------------------------------ dynamique
    def step(self, action):
        """Applique une action. Renvoie (obs, récompense, fini, info)."""
        if self.done:
            raise RuntimeError("épisode terminé : appeler reset()")
        dr, dc = MOVES[action]
        r, c = self.agent[0] + dr, self.agent[1] + dc
        if not self.grid[WALL, r, c]:
            self.agent = (r, c)
        on_lava = bool(self.grid[LAVA][self.agent])
        on_goal = self.agent == self.goal
        self.done = on_lava or on_goal
        reward = 1.0 if on_goal else (-1.0 if on_lava else -0.01)
        return self.observe(), reward, self.done, {"danger": on_lava, "success": on_goal}

    def render(self):
        """Affichage texte : # mur, ~ lave, * objectif, A agent."""
        rows = []
        for r in range(self.n):
            row = ""
            for c in range(self.n):
                if (r, c) == self.agent:
                    row += "A"
                elif self.grid[WALL, r, c]:
                    row += "#"
                elif self.grid[LAVA, r, c]:
                    row += "~"
                elif self.grid[GOAL, r, c]:
                    row += "*"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)
