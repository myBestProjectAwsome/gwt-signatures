"""Agent qui découvre ses sous-buts (feuille de route v2, lot 3).

Même agent que celui du verdict (perception, imagination, carte mentale,
valeurs verrouillées), avec une différence : la tâche ne lui donne plus que le
but final. À chaque pas, la planification par événements (event_planner.py)
choisit le sous-but qu'il vise vraiment (ex. la clé), et c'est ce sous-but qui
configure le module de coût et la carte mentale.

Ce qu'il constate l'emporte sur ce qu'il croit : quand une action ne change
rien (il s'est cogné), la case visée est notée « bloquante dans cette
situation » (avec ou sans la clé en main) pour le reste de l'épisode, et sa
carte mentale en tient compte. C'est un apprentissage de l'épisode, par la
surprise, sans aucune règle sur les portes.
"""
import numpy as np
import torch

from .environment.keydoor_world import CARRY
from .planning.event_planner import EventPlanner
from .task_agent import TaskAgent


class PrerequisiteAgent(TaskAgent):
    def __init__(self, arch, cfg, cue, novelty_sigma, effect_model, **flags):
        super().__init__(arch, cfg, cue, novelty_sigma, **flags)
        self.events = EventPlanner(arch.mental_map, effect_model)
        self.subgoal, self.subgoal_cost, self.plan_events = None, float("inf"), None
        self.blocked_seen = set()

    def reset(self, task):
        super().reset(task)
        self.subgoal, self.subgoal_cost, self.plan_events = None, float("inf"), None
        self.blocked_seen = set()

    def blocked_mask(self, obs):
        """Cases constatées bloquantes, dans la situation de cette observation (clé en main ou non)."""
        obs = torch.as_tensor(obs)
        carry = bool(obs[CARRY].max() > 0)
        m = torch.zeros(obs.shape[-2:])
        for cell, c in self.blocked_seen:
            if c == carry:
                m[cell] = 1
        return m

    def _perceive(self, obs):
        if self.prev is not None and self.last is not None and np.array_equal(self.prev, obs):
            cell = self.arch.mental_map.attempted_cell(torch.as_tensor(obs), self.last.action)
            self.blocked_seen.add((cell, bool(np.asarray(obs)[CARRY].max() > 0)))
        super()._perceive(obs)

    def act(self, obs):
        if not getattr(self, "_fresh", False):
            self._perceive(obs)
        self._fresh = True                   # le constat est fait : TaskAgent.act ne le refait pas
        if self.tracker is not None and not self.tracker.done:
            self._choose_subgoal(obs)
        return super().act(obs)

    def _choose_subgoal(self, obs):
        final = self.tracker.next_event
        seq, cost = self.events.plan(obs, final, self.blocked_mask)
        self.core.planner.blocked = self.blocked_mask(obs)
        if seq is None:                      # rien d'atteignable : on vise le but final
            seq, cost = [final], float("inf")
        self.plan_events, self.subgoal_cost = seq, cost
        new = seq[0]
        if new != self.arch.cost.target:
            self.arch.cost.configure(new)
            self.core.reset()
        self.subgoal = new
