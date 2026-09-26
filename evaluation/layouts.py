"""Familles de cartes du test final. SECRET pour l'entraînement.

Graines réservées : 7 000 000 et plus (cartes de test), 8 000 000 et plus
(cartes d'entraînement autorisées pendant la phase d'adaptation du test).
Chaque carte est retirée tant que la tâche n'y est pas soluble.
"""
import numpy as np

from mini_iag.environment.keydoor_world import (DOOR, GOAL, KEY, LAVA, WALL, KeyDoorWorld,
                                                place, standard_layout)
from mini_iag.tasks import solvable

TEST_BASE, PRACTICE_BASE = 7_000_000, 8_000_000


def enclos(world, rng):
    """Un mur traverse la carte ; une seule porte ; l'objectif est de l'autre côté."""
    world.clear()
    n, mid = world.n, world.n // 2
    vertical = bool(rng.integers(2))
    door_at = int(rng.integers(1, n - 1))
    for i in range(1, n - 1):
        cell = (i, mid) if vertical else (mid, i)
        world.grid[DOOR if i == door_at else WALL][cell] = 1
    side = lambda c: (c[1] if vertical else c[0]) < mid
    goal_side = bool(rng.integers(2))
    cells = world.free_cells()
    a_side = [c for c in cells if side(c) != goal_side]
    b_side = [c for c in cells if side(c) == goal_side]
    pick = lambda lst: lst.pop(int(rng.integers(len(lst))))
    world.grid[GOAL][pick(b_side)] = 1
    world.grid[KEY][pick(a_side)] = 1
    for _ in range(2):
        world.grid[LAVA][pick(a_side if rng.integers(2) else b_side)] = 1
    world.agent = pick(a_side)


def pieces(world, rng):
    """Pièces et couloirs : deux cloisons percées d'une ouverture chacune."""
    world.clear()
    n = world.n
    for k, line in enumerate(sorted(rng.choice(np.arange(2, n - 2), 2, replace=False))):
        vertical = bool(rng.integers(2))
        gap = int(rng.integers(1, n - 1))
        for i in range(1, n - 1):
            if i != gap:
                world.grid[WALL][(i, line) if vertical else (line, i)] = 1
    place(world, rng, LAVA, 2)
    place(world, rng, GOAL, 1)
    cells = world.free_cells()
    world.agent = cells[int(rng.integers(len(cells)))]


def lave_dense(world, rng):
    """Six cases de lave au lieu de trois."""
    world.clear()
    place(world, rng, WALL, world.cfg.n_inner_walls)
    place(world, rng, LAVA, 6)
    place(world, rng, GOAL, 1)
    cells = world.free_cells()
    world.agent = cells[int(rng.integers(len(cells)))]


LAYOUTS = {"standard": standard_layout, "enclos": enclos, "pieces": pieces,
           "lave_dense": lave_dense}


def make_map(cfg, case, index, base=TEST_BASE, case_index=0):
    """Carte n° `index` d'un cas, soluble, déterministe. Renvoie un KeyDoorWorld prêt."""
    world = KeyDoorWorld(cfg)
    layout = LAYOUTS[case.layout]
    seed = base + case_index * 10_000 + index * 100
    for attempt in range(100):
        world.reset(seed=seed + attempt, layout=layout)
        if solvable(world, case.task) and not _already_done(world, case.task):
            return world
    raise RuntimeError(f"aucune carte soluble pour {case.code} n°{index}")


def _already_done(world, task):
    return world.grid[GOAL][world.agent] > 0 and task.sequence[0] == "goal"
