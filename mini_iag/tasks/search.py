"""Recherche exacte du plus court chemin pour une tâche (utilisé par l'oracle
et par l'évaluation pour mesurer l'efficacité ; JAMAIS par l'agent).

État de recherche : (case de l'agent, clé en main ?, portes encore fermées,
clés encore au sol, progression dans la séquence de la tâche). La lave est
interdite.
"""
from collections import deque

from ..environment.gridworld import MOVES


def _successors(world, state, task):
    agent, has_key, keys, doors, prog = state
    for a, (dr, dc) in enumerate(MOVES):
        cell = (agent[0] + dr, agent[1] + dc)
        if world.grid[0][cell]:                           # mur
            continue
        if cell in doors and not has_key:
            continue
        if world.grid[1][cell]:                           # lave
            continue
        events, nk, nd, hk = [], keys, doors, has_key
        if cell in doors:
            nd = tuple(d for d in doors if d != cell)
            events.append("door")
        if cell in keys:
            nk = tuple(k for k in keys if k != cell)
            hk = True
            events.append("key")
        if world.grid[2][cell]:
            events.append("goal")
        p = prog
        for e in events:
            if p < len(task.sequence) and e == task.sequence[p]:
                p += 1
        yield a, (cell, hk, nk, nd, p)


def _bfs(world, task):
    agent, has_key, keys, doors = world.state()
    start = (agent, has_key, keys, doors, 0)
    parent = {start: None}
    q = deque([start])
    while q:
        s = q.popleft()
        if s[4] == len(task.sequence):
            return s, parent
        for a, nxt in _successors(world, s, task):
            if nxt not in parent:
                parent[nxt] = (s, a)
                q.append(nxt)
    return None, parent


def optimal_steps(world, task):
    """Nombre minimal de pas pour réussir la tâche depuis l'état actuel, ou None."""
    end, parent = _bfs(world, task)
    if end is None:
        return None
    n = 0
    while parent[end] is not None:
        end = parent[end][0]
        n += 1
    return n


def solvable(world, task):
    return optimal_steps(world, task) is not None


def oracle_action(world, task, progress=0):
    """Première action d'un plus court chemin (en tenant compte de la progression)."""
    agent, has_key, keys, doors = world.state()
    start = (agent, has_key, keys, doors, progress)
    parent = {start: None}
    q, end = deque([start]), None
    while q:
        s = q.popleft()
        if s[4] == len(task.sequence):
            end = s
            break
        for a, nxt in _successors(world, s, task):
            if nxt not in parent:
                parent[nxt] = (s, a)
                q.append(nxt)
    if end is None:
        return 0
    action = 0
    while parent[end] is not None:
        end, action = parent[end]
    return action
