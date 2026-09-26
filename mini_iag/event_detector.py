"""Détecter les événements en comparant deux observations successives.

L'agent ne reçoit que des observations : personne ne lui dit « tu as ramassé
la clé ». Il le déduit de ce qu'il voit, comme un humain :
  - clé    : le canal « clé en main » passe de 0 à 1 ;
  - porte  : une porte a disparu ;
  - objectif / lave : l'agent vient d'entrer sur une case objectif / lave.
Même ordre que le monde : porte, clé, objectif, lave.
"""
import numpy as np

from .environment.keydoor_world import AGENT, CARRY, DOOR, GOAL, LAVA


class EventDetector:
    def __call__(self, prev, obs):
        prev, obs = np.asarray(prev), np.asarray(obs)
        events = []
        if obs[DOOR].sum() < prev[DOOR].sum():
            events.append("door")
        if obs[CARRY].max() > prev[CARRY].max():
            events.append("key")
        moved = not np.array_equal(prev[AGENT], obs[AGENT])
        cell = np.unravel_index(obs[AGENT].argmax(), obs[AGENT].shape)
        if moved and obs[GOAL][cell]:
            events.append("goal")
        if obs[LAVA][cell]:
            events.append("lava")
        return events
