"""La vie continue : l'agent joue carte après carte et apprend en vivant.

Chaque épisode : une carte publique neuve (graine LIFE_BASE + n° d'épisode),
une tâche publique (au hasard, ou imposée par l'humain). Tous les
`learn_every` épisodes, une séance d'apprentissage continu.
L'état complet (imagination apprise, souvenirs récents, historique) est
sauvegardé et rechargé : éteindre l'ordinateur ne fait rien oublier.

Tâches de la vie : les tâches publiques uniquement (jamais celles du test).

La critique d'actions apprend aussi en vivant (learn_critic, diagnostic 10).

Glace (n_ice > 0, désactivée par défaut) : une règle absente de toute
l'expérience d'origine, pour vérifier que l'agent peut la découvrir en
vivant (diagnostic 11). La perception gagne un sens (grow_senses).

Exploration libre : une part des épisodes (explore_fraction) n'a AUCUNE tâche.
L'agent se promène, guidé seulement par la curiosité (la mémoire le pousse vers
les endroits nouveaux) et la peur de la lave. Il ramasse des clés, se promène
avec, ouvre des portes... sans qu'aucune tâche ne le lui demande. Sans cela,
la vie ne montre presque jamais « avoir la clé en main » (les tâches s'arrêtent
au ramassage). Mesuré (diagnostic 9) : elle double les portes ouvertes vécues,
mais ne corrige pas l'oubli de la porte. Désactivée par défaut (0.0) : elle ne
change rien de mesurable (« si le retirer ne change rien, c'est décoratif »).
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
EXPLORATION = "exploration libre"   # nom des épisodes sans tâche dans l'historique


def anchor_segments(cfg, n_maps=2000):
    """Souvenirs anciens : un échantillon fixe de l'expérience d'origine (étape 4a)."""
    return collect_kd_segments(cfg, n_maps=n_maps, per_map=5, horizon=cfg.planning_horizon,
                               seed=TRAIN_SEED)


class Life:
    def __init__(self, agent, learn=True, learn_every=5, old_fraction=0.5, max_steps=40,
                 path=LIFE_PATH, seed=0, rare_bonus="défaut", explore_fraction=0.0,
                 explore_steps=30, tasks=LIFE_TASKS, learn_critic=True, n_ice=0):
        self.agent, self.cfg = agent, agent.cfg
        self.learn, self.learn_every, self.max_steps = learn, learn_every, max_steps
        self.explore_fraction, self.explore_steps = explore_fraction, explore_steps
        self.path = Path(path)
        if n_ice and getattr(agent.arch.world_model, "new_senses", None) is None:
            agent.arch.world_model.grow_senses(1)        # la glace : un nouveau sens
        extra = {} if rare_bonus == "défaut" else {"rare_bonus": rare_bonus}
        self.buffer = ExperienceBuffer(anchor_segments(self.cfg), horizon=self.cfg.planning_horizon,
                                       seed=seed, **extra)
        self.learner = ContinualLearner(agent.arch, self.buffer, old_fraction=old_fraction,
                                        learn_critic=learn_critic)
        self.tasks = tuple(tasks)
        self.world = KeyDoorWorld(self.cfg, n_ice=n_ice)
        self.rng = random.Random(seed)
        self.episode, self.history, self.forced_task = 0, [], None

    # ------------------------------------------------------------ un épisode
    def choose_task(self):
        """Une tâche publique, ou None (exploration libre) avec la probabilité explore_fraction."""
        if self.forced_task:
            return self.forced_task
        if self.rng.random() < self.explore_fraction:
            return None
        return self.rng.choice(self.tasks)

    def live_one(self, on_step=None):
        task = self.choose_task()
        seed = LIFE_BASE + self.episode * 10
        for attempt in range(10):                  # carte soluble pour cette tâche
            self.world.reset(seed=seed + attempt)
            if task is None or solvable(self.world, task):
                break
        steps_max = self.max_steps if task is not None else self.explore_steps
        outcome, steps, traj = run_episode(self.agent, self.world, task, steps_max, on_step)
        surprise = self.learner.surprise(traj)
        self.buffer.add(traj)
        self.episode += 1
        loss = None
        if self.learn and self.episode % self.learn_every == 0:
            loss = self.learner.update()
        record = {"episode": self.episode, "task": task.name if task else EXPLORATION,
                  "outcome": outcome,
                  "steps": steps, "surprise": surprise, "loss": loss, "time": time.time()}
        self.history.append(record)
        return record

    # ------------------------------------------------------------ bilan
    def recent(self, n=100, task=None):
        h = [r for r in self.history
             if r["task"] != EXPLORATION and task in (None, r["task"])][-n:]
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
                    "history": self.history, "n_updates": self.learner.n_updates,
                    "senses": ({"encoder": self.agent.arch.world_model.encoder.net[0].state_dict(),
                                "target": self.agent.arch.world_model.target_encoder.net[0].state_dict()}
                               if getattr(self.agent.arch.world_model, "new_senses", None) else None),
                    "critic": (self.learner.critic_learner.state()
                               if self.learner.critic_learner else None)}, self.path)

    def load(self):
        if not self.path.exists():
            return False
        state = torch.load(self.path, weights_only=False)
        if state.get("senses"):
            wm = self.agent.arch.world_model
            if getattr(wm, "new_senses", None) is None:
                wm.grow_senses(1)
            wm.encoder.net[0].load_state_dict(state["senses"]["encoder"])
            wm.target_encoder.net[0].load_state_dict(state["senses"]["target"])
        self.agent.arch.world_model.predictor.load_state_dict(state["predictor"])
        self.learner.opt.load_state_dict(state["optimizer"])
        self.buffer.load_state(state["buffer"])
        self.episode, self.history = state["episode"], state["history"]
        self.learner.n_updates = state["n_updates"]
        if self.learner.critic_learner is not None and state.get("critic"):
            self.learner.critic_learner.load_state(state["critic"])
        return True
