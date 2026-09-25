"""Étape 3 : le câblage.  Lancer :  python -m mini_iag.coordinate

  1. part des modules éduqués à l'étape 2 (checkpoints/step2.pt) ;
  2. affinage de coordination : le coût apprend à évaluer des états IMAGINÉS ;
  3. calibre la mémoire (échelle de distance entre deux états voisins) ;
  4. teste l'agent complet sur 50 cartes jamais vues.
Sauvegarde : checkpoints/step3.pt
"""
import argparse
import time
from pathlib import Path

import torch

from .agent import Agent
from .config import Config
from .data import TEST_SEED, TRAIN_SEED, collect, collect_segments
from .environment import GridWorld
from .train import TRAIN_MAPS, load, train
from .training import CoordinationTrainer

STEP2, STEP3 = Path("checkpoints/step2.pt"), Path("checkpoints/step3.pt")
DEMO_OFFSET = 5000          # cartes de test réservées à l'évaluation des agents


@torch.no_grad()
def novelty_scale(world_model, cfg):
    """Distance typique entre les latents de deux états voisins (l'agent a bougé)."""
    d = collect(cfg, n_maps=200, seed=TRAIN_SEED)
    moved = (d.obs[:, 3] != d.next_obs[:, 3]).flatten(1).any(1)
    z0, z1 = world_model.encode(d.obs[moved]), world_model.encode(d.next_obs[moved])
    return float((z1 - z0).norm(dim=1).median())


def run_episode(agent, env, map_seed, max_steps):
    obs = env.reset(seed=map_seed)
    optimal = env.distance_to_goal()
    agent.reset()
    agree = []
    for t in range(1, max_steps + 1):
        dec = agent.act(obs)
        agree.append(dec.agrees)
        obs, _, done, info = env.step(dec.action)
        if done:
            break
    outcome = "succès" if info["success"] else ("lave" if info["danger"] else "bloqué")
    return {"outcome": outcome, "steps": t, "optimal": optimal,
            "agreement": sum(agree) / len(agree)}


def coordinate(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    if not STEP2.exists():
        say("Pas de checkpoint de l'étape 2 : entraînement (python -m mini_iag.train)...")
        train(STEP2, verbose=verbose)
    ckpt = torch.load(STEP2, weights_only=False)
    arch, readout, _ = load(STEP2)
    cfg = arch.cfg
    t0 = time.time()

    say(f"Affinage de coordination : le coût apprend les états imaginés "
        f"(jusqu'à {cfg.planning_horizon} pas)")
    segments = collect_segments(cfg, n_maps=TRAIN_MAPS, horizon=cfg.planning_horizon,
                                seed=TRAIN_SEED)
    log = CoordinationTrainer(arch.cost, arch.world_model).fit(segments, verbose=verbose)
    sigma = novelty_scale(arch.world_model, cfg)
    say(f"\nMémoire : distance typique entre deux états voisins = {sigma:.2f}")

    STEP3.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ckpt, "architecture": arch.state_dict(), "novelty_sigma": sigma,
                "coordination_log": log}, STEP3)

    say("\nTest rapide de l'agent complet sur 50 cartes jamais vues...")
    agent = Agent(arch, cfg, ckpt["cue"], sigma)
    env = GridWorld(cfg)
    res = [run_episode(agent, env, TEST_SEED * 10_000 + DEMO_OFFSET + m, cfg.max_steps)
           for m in range(50)]
    for k in ("succès", "lave", "bloqué"):
        say(f"  {k:7} {100 * sum(r['outcome'] == k for r in res) / len(res):5.1f} %")
    say(f"\nTerminé en {time.time() - t0:.0f} s. Sauvegardé dans {STEP3}")


def load_agent(path=STEP3, **flags):
    """Charge l'agent de l'étape 3. flags : use_memory, use_workspace, ..."""
    ckpt = torch.load(path, weights_only=False)
    arch, _, _ = load(path)
    return Agent(arch, Config(**ckpt["config"]), ckpt["cue"], ckpt["novelty_sigma"], **flags)


if __name__ == "__main__":
    argparse.ArgumentParser(prog="python -m mini_iag.coordinate").parse_args()
    coordinate()
