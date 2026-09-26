"""Étape 4a : éduquer la mini-IAG dans le monde v2 (clé, porte).

    python -m mini_iag.train_keydoor        (~5 minutes sur CPU)

  1. modèle du monde : apprend la PHYSIQUE (murs, lave, clé, porte) par
     l'expérience d'un agent qui marche au hasard, sans aucune récompense ;
  2. module de coût configurable : apprend à reconnaître les 4 événements
     (objectif, clé, porte, lave) dans des états réels ET imaginés ;
  3. critique : proximité de chaque événement, apprise par itération de valeur
     dans l'imagination (sert au-delà de l'horizon de planification) ;
  4. espace de travail : attention sélective, comme à l'étape 2 ;
  5. mémoire : échelle de distance entre deux états voisins.
Aucune tâche n'est utilisée ici : les tâches n'interviennent qu'au moment
d'agir, par le configurateur.
Sauvegarde : checkpoints/step4.pt
"""
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .architecture_v2 import ArchitectureV2
from .config import Config
from .data import TEST_SEED, TRAIN_SEED
from .data_keydoor import collect_kd, collect_kd_segments
from .training import (CoordinationTrainer, CriticTrainer, SelectionReadout, WorkspaceTrainer,
                       WorldModelTrainer)

STEP4 = Path("checkpoints/step4.pt")
TRAIN_MAPS = 3000
EVENT_POS_WEIGHT = (10.0, 10.0, 30.0, 1.0)     # objectif, clé, porte (très rare), lave


@torch.no_grad()
def novelty_scale(world_model, data, n=20000):
    obs, nxt = data.obs[:n].float(), data.next_obs[:n].float()
    moved = (obs[:, 3] != nxt[:, 3]).flatten(1).any(1)
    return float((world_model.encode(nxt[moved]) - world_model.encode(obs[moved])).norm(dim=1).median())


def train(out=STEP4, verbose=True, **overrides):
    say = print if verbose else (lambda *a, **k: None)
    torch.manual_seed(0)
    cfg = Config.keydoor(**overrides)
    arch = ArchitectureV2(cfg)
    t0 = time.time()
    say(f"Collecte : {TRAIN_MAPS} cartes, agent qui marche au hasard...")
    data = collect_kd(cfg, n_maps=TRAIN_MAPS, seed=TRAIN_SEED)
    test = collect_kd(cfg, n_maps=200, steps=40, seed=TEST_SEED)
    segments = collect_kd_segments(cfg, n_maps=TRAIN_MAPS // 2, horizon=cfg.multistep_horizon,
                                   seed=TRAIN_SEED)
    say(f"  {len(data)} transitions, {len(segments)} segments de {cfg.multistep_horizon} pas\n")

    say("1/4 Modèle du monde (physique du monde v2)")
    arch.world_model.event_pos_weight = torch.tensor(EVENT_POS_WEIGHT)
    wm_log = WorldModelTrainer(arch.world_model, iters=8000).fit(data, test, segments,
                                                                verbose=verbose)
    for p in arch.world_model.parameters():
        p.requires_grad = False

    say("\n2/4 Module de coût configurable (4 événements, transitions réelles et imaginées)")
    coord_segments = collect_kd_segments(cfg, n_maps=TRAIN_MAPS // 2, horizon=cfg.planning_horizon,
                                         seed=TRAIN_SEED)
    cost_log = CoordinationTrainer(arch.cost, arch.world_model, iters=4000).fit(
        coord_segments, verbose=verbose, eval_every=1000)

    say("\n3/4 Critique (itération de valeur dans l'imagination)")
    with torch.no_grad():
        latents = arch.world_model.encode(data.obs[:60000].float())
    critic_log = CriticTrainer(arch.critic, arch.world_model, arch.cost,
                               discount=cfg.value_discount).fit(latents, verbose=verbose)

    say("\n4/4 Espace de travail")
    readout = SelectionReadout(cfg)
    ws = WorkspaceTrainer(arch.workspace, readout, arch.world_model)
    ws_log = ws.fit(data.next_obs[:40000].float(), test.next_obs.float(), verbose=verbose,
                    eval_every=1000)
    sigma = novelty_scale(arch.world_model, data)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "architecture": arch.state_dict(),
                "readout": readout.state_dict(), "cue": ws.cue, "novelty_sigma": sigma,
                "log": {"world_model": wm_log, "cost": cost_log, "critic": critic_log,
                        "workspace": ws_log}}, out)
    say(f"\nMémoire : distance typique entre deux états voisins = {sigma:.2f}")
    say(f"Terminé en {time.time() - t0:.0f} s. Sauvegardé dans {out}")


def load_task_agent(path=STEP4, cfg_overrides=None, **flags):
    from .task_agent import TaskAgent
    ckpt = torch.load(path, weights_only=False)
    cfg = Config(**{**ckpt["config"], **(cfg_overrides or {})})
    arch = ArchitectureV2(cfg)
    arch.load_state_dict(ckpt["architecture"])
    arch.eval()
    return TaskAgent(arch, cfg, ckpt["cue"], ckpt["novelty_sigma"], **flags)


if __name__ == "__main__":
    train()
