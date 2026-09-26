"""Agent à tâches (monde v2) : il reçoit une tâche et la réalise sous-but par sous-but.

Une tâche est une séquence d'événements (ex. objectif, puis clé). L'agent :
  1. ne sait que ce qu'il VOIT : il détecte les événements en comparant ses
     observations successives (EventDetector) et suit sa progression ;
  2. transmet au module de coût le prochain événement à provoquer
     (le configurateur) : le coût change, pas les connaissances ;
  3. planifie comme à l'étape 3 (imaginer, évaluer, choisir) ;
  4. quand un sous-but est atteint, vide sa mémoire des lieux visités : repasser
     par un endroit connu est normal quand le but change ;
  5. apprend de ses surprises : si une action ne change RIEN à ce qu'il voit
     (mur, porte fermée), il note qu'elle est inutile dans cette situation et
     ne la retente pas. Sans cela, il pousse indéfiniment contre un mur que son
     modèle du monde croit franchissable (mesuré : jusqu'à 20 fois de suite).

Interface attendue par le test final : reset(task), act(obs), cost_module.
"""
import numpy as np
import torch

from .agent import Agent
from .event_detector import EventDetector
from .tasks import TaskTracker


class TaskAgent:
    def __init__(self, arch, cfg, cue, novelty_sigma, **flags):
        self.core = Agent(arch, cfg, cue, novelty_sigma,
                          **{k: v for k, v in flags.items() if k != "no_surprise"})
        self.arch, self.cfg = arch, cfg
        self.detect = EventDetector()
        self.tracker, self.prev, self.last = None, None, None
        self.use_surprise = flags.get("use_memory", True) and not flags.get("no_surprise", False)
        self.useless = {}          # situation (observation) -> actions constatées inutiles

    @property
    def cost_module(self):
        return self.arch.cost

    def reset(self, task):
        self.tracker = TaskTracker(task)
        self.prev, self.last = None, None
        self.useless = {}
        self.core.reset()
        self.arch.cost.configure(self.tracker.next_event)

    @torch.no_grad()
    def act(self, obs):
        if self.prev is not None:
            if self.use_surprise and np.array_equal(self.prev, obs):
                self.useless.setdefault(obs.tobytes(), set()).add(self.last.action)
            before = self.tracker.progress
            self.tracker.update(self.detect(self.prev, obs))
            if self.tracker.progress != before and not self.tracker.done:
                self.arch.cost.configure(self.tracker.next_event)
                self.core.reset()
        self.last = self.core.act(obs, tuple(self.useless.get(np.asarray(obs).tobytes(), ())))
        self.prev = obs
        return self.last.action
