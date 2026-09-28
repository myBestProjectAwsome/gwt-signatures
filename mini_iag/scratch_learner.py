"""Référence « repartant de zéro » (feuille de route v2, lot 1).

La référence sans connaissances du premier test (build_relearner) était trop
faible : ses modules aléatoires n'étaient pas tous entraînables par practice()
(la carte mentale, la perception et l'évaluation restaient aléatoires).

Celle-ci repart de zéro mais peut TOUT apprendre : à chaque practice(), elle
joue les épisodes d'adaptation avec son propre comportement, garde toute
l'expérience accumulée, puis réentraîne TOUS ses modules sur cette expérience
seulement, avec la même procédure que l'agent (modèle du monde, module de coût,
critique d'actions, espace de travail, carte mentale).

Ce qui la sépare de l'agent : la QUANTITÉ d'expérience. L'agent a été
préentraîné sur 150 000 transitions (3 000 cartes au hasard), elle n'a que les
épisodes d'adaptation. La comparaison mesure donc le bénéfice des acquis
antérieurs. Les budgets sont rapportés séparément (budget()).
"""
import numpy as np
import torch

from .architecture_v2 import ArchitectureV2
from .config import Config
from .data import Segments, Transitions
from .data_keydoor import encode_events
from .life.episode import run_episode
from .modules.mental_map import MentalMap
from .task_agent import TaskAgent
from .train_keydoor import EVENT_POS_WEIGHT, novelty_scale
from .train_map import PARAMS
from .training import (CoordinationTrainer, MapTrainer, OfflineQTrainer, SelectionReadout,
                       WorkspaceTrainer, WorldModelTrainer)

FULL_ITERS = {"world_model": 8000, "cost": 4000, "critic": 30000, "workspace": 3000, "map": 12000}


def segments_from(episodes, horizon):
    """Tous les morceaux de H pas des trajectoires vécues (pas masqués après la fin)."""
    obs0, acts, vis, evs, alv = [], [], [], [], []
    for ep in episodes:
        T = len(ep["action"])
        for s in range(T):
            idx = np.arange(s, min(s + horizon, T))
            pad = horizon - len(idx)
            acts.append(np.concatenate([ep["action"][idx], np.zeros(pad, int)]))
            vis.append(np.concatenate([ep["next_obs"][idx], np.repeat(ep["next_obs"][idx[-1:]], pad, 0)]))
            evs.append(np.concatenate([ep["events"][idx], np.zeros((pad, ep["events"].shape[1]))]))
            alv.append(np.concatenate([np.ones(len(idx), bool), np.zeros(pad, bool)]))
            obs0.append(ep["obs"][s])
    t = lambda x, dt=None: torch.tensor(np.array(x), dtype=dt)
    return Segments(t(obs0, torch.float32), t(acts), t(vis, torch.float32), t(evs, torch.float32), t(alv))


def transitions_from(episodes):
    obs = np.concatenate([e["obs"] for e in episodes])
    tr = Transitions(torch.tensor(obs), torch.tensor(np.concatenate([e["action"] for e in episodes])),
                     torch.tensor(np.concatenate([e["next_obs"] for e in episodes])))
    tr._events = torch.tensor(np.concatenate([e["events"] for e in episodes]), dtype=torch.float32)
    return tr


class ScratchLearner(TaskAgent):
    def __init__(self, seed=0, iters=None, cfg=None, novelty_sigma=1.0):
        cfg = cfg or Config.keydoor()
        torch.manual_seed(seed)
        arch = ArchitectureV2(cfg)
        arch.mental_map = MentalMap(cfg, **PARAMS)
        arch.eval()
        super().__init__(arch, cfg, torch.randn(cfg.latent_dim), novelty_sigma)
        self.iters = {**FULL_ITERS, **(iters or {})}
        self.seed, self.episodes, self.n_trainings = seed, [], 0
        self.cost_module.lock()

    def budget(self):
        steps = int(sum(len(e["action"]) for e in self.episodes))
        return {"épisodes vécus": len(self.episodes), "transitions": steps,
                "itérations par entraînement": self.iters, "entraînements": self.n_trainings}

    def practice(self, make_world, task, episodes, max_steps=50):
        for i in range(episodes):
            _, _, traj = run_episode(self, make_world(i), task, max_steps)
            if len(traj["action"]):
                self.episodes.append(traj)
        self.retrain()

    def retrain(self, verbose=False):
        """Réentraîne tous les modules, à partir de zéro, sur toute l'expérience vécue."""
        cfg, it = self.cfg, self.iters
        torch.manual_seed(self.seed + 1000 * (self.n_trainings + 1))
        arch = ArchitectureV2(cfg)
        data = transitions_from(self.episodes)
        wm_seg = segments_from(self.episodes, cfg.multistep_horizon)
        plan_seg = segments_from(self.episodes, cfg.planning_horizon)
        arch.world_model.event_pos_weight = torch.tensor(EVENT_POS_WEIGHT)
        WorldModelTrainer(arch.world_model, iters=it["world_model"]).fit(data, None, wm_seg, verbose=verbose)
        for p in arch.world_model.parameters():
            p.requires_grad = False
        CoordinationTrainer(arch.cost, arch.world_model, iters=it["cost"]).fit(plan_seg, verbose=verbose)
        OfflineQTrainer(arch.critic, arch.world_model, discount=cfg.value_discount,
                        iters=it["critic"]).fit(data, verbose=verbose)
        readout = SelectionReadout(cfg)
        ws = WorkspaceTrainer(arch.workspace, readout, arch.world_model, iters=it["workspace"])
        ws.fit(data.next_obs.float(), verbose=verbose)
        arch.mental_map = MentalMap(cfg, **PARAMS)
        MapTrainer(arch.mental_map, iters=it["map"], cell_weight=1.0).fit(data, verbose=verbose)
        arch.eval()
        arch.cost.lock()
        sigma = novelty_scale(arch.world_model, data)
        self.__init_agent(arch, ws.cue, sigma)
        self.n_trainings += 1

    def __init_agent(self, arch, cue, sigma):
        episodes, iters, seed, n = self.episodes, self.iters, self.seed, self.n_trainings
        TaskAgent.__init__(self, arch, self.cfg, cue, sigma)
        self.episodes, self.iters, self.seed, self.n_trainings = episodes, iters, seed, n
