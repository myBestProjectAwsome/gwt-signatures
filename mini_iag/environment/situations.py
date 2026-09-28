"""Quatre situations de prérequis (feuille de route v2, lot 2). Matériel PUBLIC de développement.

Dans chaque situation, on ne demande QUE le but final (l'objectif). C'est à
l'agent de découvrir s'il a besoin de la clé :

  accessible       l'objectif est atteignable directement      -> y aller directement
  incontournable   l'objectif est derrière une porte fermée    -> prendre la clé, puis passer
  ouverte          le passage est déjà ouvert, une clé traîne  -> ignorer la clé inutile
  detour           une porte ferme le raccourci, mais un détour
                   sans clé est plus court que d'aller la chercher -> prendre le détour

Chaque carte est VÉRIFIÉE par une recherche exacte (search.py, l'oracle réservé
au diagnostic) : la situation voulue est bien celle de la carte. Les positions
et l'orientation du mur varient à chaque carte (pas de repère fixe).
"""
import copy

import numpy as np

from ..tasks import Task, optimal_steps
from .keydoor_world import DOOR, GOAL, KEY, LAVA, WALL

GOAL_ONLY = Task("objectif", ("goal",), "Atteins l'objectif (*).")
KEY_ONLY = Task("clé", ("key",), "")
SITUATIONS = ("accessible", "incontournable", "ouverte", "detour")


def _dist(world, remove=(), task=GOAL_ONLY):
    w = copy.deepcopy(world)
    for ch in remove:
        w.grid[ch] = 0
    return optimal_steps(w, task)


def _via_key(world):
    """Longueur du chemin qui passe d'abord par la clé, puis va à l'objectif."""
    d1 = optimal_steps(world, KEY_ONLY)
    if d1 is None:
        return None
    w = copy.deepcopy(world)
    cell = tuple(np.argwhere(w.grid[KEY] > 0)[0])
    w.grid[KEY][cell] = 0
    w.agent, w.has_key = cell, True
    d2 = 0 if w.grid[GOAL][cell] else optimal_steps(w, GOAL_ONLY)
    return None if d2 is None else d1 + d2


def classify(world):
    """Situation réelle d'une carte (ou None), d'après les plus courts chemins exacts."""
    d = _dist(world)                               # avec tout
    if d is None:
        return None
    no_key = _dist(world, remove=(KEY,))           # sans pouvoir prendre la clé
    door_open = _dist(world, remove=(DOOR,))       # si la porte était déjà ouverte
    has_door = world.grid[DOOR].any()
    if no_key is None:
        return "incontournable"
    if not has_door:
        return "ouverte"
    via_key = _via_key(world)
    if door_open is not None and door_open < no_key and via_key is not None and no_key < via_key:
        return "detour"                            # la porte serait un raccourci, la clé coûte trop
    if door_open == no_key:
        return "accessible"                        # la porte ne sert à rien
    return None


def _line(world, rng, gaps):
    """Un mur qui traverse la carte (vertical ou horizontal), percé de `gaps` ouvertures."""
    n = world.n
    pos = int(rng.integers(2, n - 2))
    cells = [(r, pos) for r in range(1, n - 1)]
    if rng.integers(2):
        cells = [(c, r) for r, c in cells]
    for cell in cells:
        world.grid[WALL][cell] = 1
    holes = [cells[i] for i in rng.permutation(len(cells))[:gaps]]
    for cell in holes:
        world.grid[WALL][cell] = 0
    vertical = cells[0][1] == cells[1][1]
    side = (lambda rc: rc[1] < pos) if vertical else (lambda rc: rc[0] < pos)
    return holes, side


def _free(world, pred=lambda rc: True):
    return [rc for rc in world.free_cells() if pred(rc)]


def _put(world, rng, channel, cells):
    cell = cells[int(rng.integers(len(cells)))]
    world.grid[channel][cell] = 1
    return cell


def build(world, rng, situation):
    """Construit une carte candidate de la situation demandée (non vérifiée)."""
    world.clear()
    world.has_key, world.dead = False, False
    if situation == "accessible":
        for ch, k in ((WALL, world.cfg.n_inner_walls), (LAVA, 2), (GOAL, 1), (KEY, 1), (DOOR, 1)):
            for _ in range(k):
                _put(world, rng, ch, _free(world))
    else:
        holes, near = _line(world, rng, 2 if situation == "detour" else 1)
        if situation in ("incontournable", "detour"):
            world.grid[DOOR][holes[0]] = 1
        far = lambda rc: not near(rc)
        _put(world, rng, GOAL, _free(world, far))
        _put(world, rng, KEY, _free(world, near))
        for _ in range(int(rng.integers(0, 3))):
            _put(world, rng, LAVA, _free(world))
        world.agent = None
        cells = _free(world, near)
        world.agent = cells[int(rng.integers(len(cells)))]
        return
    cells = _free(world)
    world.agent = cells[int(rng.integers(len(cells)))]


def make_situation(world, situation, seed, tries=500):
    """Carte de la situation demandée, vérifiée par l'oracle. Renvoie la graine utilisée."""
    for t in range(tries):
        rng = np.random.default_rng(seed * 1000 + t)
        build(world, rng, situation)
        if world.agent is not None and classify(world) == situation:
            return seed * 1000 + t
    raise RuntimeError(f"pas de carte « {situation} » trouvée")
