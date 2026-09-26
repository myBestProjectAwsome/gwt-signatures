"""La vie continue : l'agent joue carte après carte et apprend en vivant.

Chaque épisode : une carte publique neuve (graine LIFE_BASE + n° d'épisode),
une tâche publique (au hasard, ou imposée par l'humain). Tous les
`learn_every` épisodes, une séance d'apprentissage continu.
L'état complet (imagination apprise, souvenirs récents, historique) est
sauvegardé et rechargé : éteindre l'ordinateur ne fait rien oublier.

Tâches de la vie : les tâches publiques uniquement (jamais celles du test).
"""
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data import TRAIN_SEED
from ..data_keydoor import collect_kd_segments
from ..environment.keydoor_world import KeyDoorWorld
from ..tasks import TRAINING_TASKS, Task, solvable
from .continual_learner import ContinualLearner
from .episode import run_episode
from .experience_buffer import ExperienceBuffer

LIFE_BASE = 1_000_000          # graines des cartes de la vie (publiques, jamais le test)
LIFE_TASKS = TRAINING_TASKS + (Task("objectif puis clé", ("goal", "key"),
                                    "Va sur l'objectif, PUIS ramasse la clé."),)
LIFE_PATH = Path("checkpoints/life.pt")


def anchor_segments(cfg, n_maps=1000):
    """Souvenirs anciens : un échantillon fixe de l'expérience d'origine (étape 4a)."""
    return collect_kd_segments(cfg, n_maps=n_maps, per_map=5, horizon=cfg.planning_horizon,
                               seed=TRAIN_SEED)


class Life:
    def __init__(self, agent, learn=True, learn_every=5, old_fraction=0.5, max_steps=40,
                 path=LIFE_PATH, seed=0):
        self.agent, self.cfg = agent, agent.cfg
        self.learn, self.learn_every, self.max_steps = learn, learn_every, max_steps
        self.path = Path(path)
        self.buffer = ExperienceBuffer(anchor_segments(self.cfg), horizon=self.cfg.planning_horizon,
                                       seed=seed)
        self.learner = ContinualLearner(agent.arch, self.buffer, old_fraction=old_fraction)
        self.world = KeyDoorWorld(self.cfg)
        self.rng = random.Random(seed)
        self.episode, self.history, self.forced_task = 0, [], None

    # ------------------------------------------------------------ un épisode
    def choose_task(self):
        return self.forced_task or self.rng.choice(LIFE_TASKS)

    def live_one(self, on_step=None):
        task = self.choose_task()
        seed = LIFE_BASE + self.episode * 10
        for attempt in range(10):                  # carte soluble pour cette tâche
            self.world.reset(seed=seed + attempt)
            if solvable(self.world, task):
                break
        outcome, steps, traj = run_episode(self.agent, self.world, task, self.max_steps, on_step)
        surprise = self.learner.surprise(traj)
        self.buffer.add(traj)
        self.episode += 1
        loss = None
        if self.learn and self.episode % self.learn_every == 0:
            loss = self.learner.update()
        record = {"episode": self.episode, "task": task.name, "outcome": outcome,
                  "steps": steps, "surprise": surprise, "loss": loss, "time": time.time()}
        self.history.append(record)
        return record

    # ------------------------------------------------------------ bilan
    def recent(self, n=100, task=None):
        h = [r for r in self.history if task in (None, r["task"])][-n:]
        if not h:
            return None
        return {k: 100 * float(np.mean([r["outcome"] == k for r in h]))
                for k in ("succès", "lave", "bloqué")} | {"n": len(h)}

    # ------------------------------------------------------------ sauvegarde
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"predictor": self.agent.arch.world_model.predictor.state_dict(),
                    "optimizer": self.learner.opt.state_dict(),
                    "buffer": self.buffer.state(), "episode": self.episode,
                    "history": self.history, "n_updates": self.learner.n_updates}, self.path)

    def load(self):
        if not self.path.exists():
            return False
        state = torch.load(self.path, weights_only=False)
        self.agent.arch.world_model.predictor.load_state_dict(state["predictor"])
        self.learner.opt.load_state_dict(state["optimizer"])
        self.buffer.load_state(state["buffer"])
        self.episode, self.history = state["episode"], state["history"]
        self.learner.n_updates = state["n_updates"]
        return True
