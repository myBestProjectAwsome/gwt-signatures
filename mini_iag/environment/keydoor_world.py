"""Monde v2 : clés et portes (étape 4). La PHYSIQUE de ce monde est publique.

Objets : murs, lave, objectif, clé, porte, agent.
  - marcher sur la clé la ramasse (événement "key") ; l'agent la garde ;
  - une porte bloque comme un mur, SAUF si l'agent a la clé : il entre alors
    dans la case, la porte s'ouvre et disparaît (événement "door") ;
  - marcher sur l'objectif déclenche l'événement "goal" ;
  - marcher sur la lave déclenche "lava" et termine l'épisode.

Contrairement au monde v1, atteindre l'objectif NE termine PAS l'épisode :
c'est la TÂCHE en cours (mini_iag/tasks) qui décide quand elle est réussie.
Le même monde sert ainsi à plusieurs tâches différentes.

Observation : tableau (7, n, n) de 0/1
    0 mur, 1 lave, 2 objectif, 3 agent, 4 clé, 5 porte, 6 "clé en main"
    (le canal 6 est rempli de 1 quand l'agent porte la clé).

Ce qui est SECRET (evaluation/) : les tâches demandées le jour du test et
certaines familles de cartes. Ce fichier ne contient que les cartes
d'entraînement "standard".
"""
import numpy as np

from .gridworld import MOVES

WALL, LAVA, GOAL, AGENT, KEY, DOOR, CARRY = range(7)
N_CHANNELS = 7
EVENTS = ("goal", "key", "door", "lava")


class KeyDoorWorld:
    def __init__(self, cfg, seed=0, n_keys=1, n_doors=1):
        self.cfg, self.n = cfg, cfg.grid_size
        self.n_keys, self.n_doors = n_keys, n_doors
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------ cartes
    def reset(self, seed=None, layout=None):
        """Nouvelle carte. `layout` : fonction (monde, rng) qui place les objets
        (sert aux familles de cartes de l'évaluation). Par défaut : carte standard."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        (layout or standard_layout)(self, self.rng)
        self.has_key, self.dead = False, False
        return self.observe()

    def clear(self):
        n = self.n
        self.grid = np.zeros((6, n, n), dtype=np.float32)    # mur, lave, objectif, -, clé, porte
        self.grid[WALL, 0, :] = self.grid[WALL, -1, :] = 1
        self.grid[WALL, :, 0] = self.grid[WALL, :, -1] = 1

    def free_cells(self):
        n = self.n
        return [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                if not self.grid[[WALL, LAVA, GOAL, KEY, DOOR], r, c].any()]

    def observe(self):
        obs = np.zeros((N_CHANNELS, self.n, self.n), dtype=np.float32)
        obs[[WALL, LAVA, GOAL, KEY, DOOR]] = self.grid[[WALL, LAVA, GOAL, KEY, DOOR]]
        obs[AGENT][self.agent] = 1
        if self.has_key:
            obs[CARRY] = 1
        return obs

    # ------------------------------------------------------------ physique
    def passable(self, cell, has_key):
        if self.grid[WALL][cell]:
            return False
        if self.grid[DOOR][cell]:
            return has_key
        return True

    def step(self, action):
        """Renvoie (obs, événements de ce pas, mort ?)."""
        if self.dead:
            raise RuntimeError("agent mort : appeler reset()")
        dr, dc = MOVES[action]
        cell = (self.agent[0] + dr, self.agent[1] + dc)
        events = []
        if self.passable(cell, self.has_key):
            self.agent = cell
            if self.grid[DOOR][cell]:
                self.grid[DOOR][cell] = 0
                events.append("door")
            if self.grid[KEY][cell]:
                self.grid[KEY][cell] = 0
                self.has_key = True
                events.append("key")
            if self.grid[GOAL][cell]:
                events.append("goal")
            if self.grid[LAVA][cell]:
                events.append("lava")
                self.dead = True
        return self.observe(), events, self.dead

    # ------------------------------------------------------------ état (pour la recherche)
    def state(self):
        doors = tuple(map(tuple, np.argwhere(self.grid[DOOR] > 0)))
        keys = tuple(map(tuple, np.argwhere(self.grid[KEY] > 0)))
        return self.agent, self.has_key, keys, doors

    def render(self):
        """# mur  ~ lave  * objectif  k clé  D porte  A agent (a = agent avec la clé)."""
        rows = []
        for r in range(self.n):
            row = ""
            for c in range(self.n):
                if (r, c) == self.agent:
                    row += "a" if self.has_key else "A"
                elif self.grid[WALL, r, c]:
                    row += "#"
                elif self.grid[DOOR, r, c]:
                    row += "D"
                elif self.grid[LAVA, r, c]:
                    row += "~"
                elif self.grid[KEY, r, c]:
                    row += "k"
                elif self.grid[GOAL, r, c]:
                    row += "*"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)


def place(world, rng, channel, count):
    cells = world.free_cells()
    for i in rng.permutation(len(cells))[:count]:
        world.grid[channel][cells[i]] = 1


def standard_layout(world, rng):
    """Carte d'entraînement : murs, lave, objectif, clé et porte placés au hasard."""
    world.clear()
    place(world, rng, WALL, world.cfg.n_inner_walls)
    place(world, rng, LAVA, world.cfg.n_lava)
    place(world, rng, GOAL, 1)
    place(world, rng, KEY, world.n_keys)
    place(world, rng, DOOR, world.n_doors)
    cells = world.free_cells()
    world.agent = cells[rng.integers(len(cells))]
