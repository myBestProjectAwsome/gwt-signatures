"""Étape 2 : l'éducation des modules.  Lancer :  python -m mini_iag.train

Ordre (cf. README) :
  1. le modèle du monde, auto-supervisé, sur des transitions ;
  2. le module de coût, SUR les latents du modèle du monde (gelé) ;
  3. l'espace de travail, SUR ces mêmes latents.
Tout est entraîné sur les cartes de graine 0 ; les mesures affichées pendant
l'entraînement portent sur les cartes de graine 1, jamais vues.

Sauvegarde : checkpoints/step2.pt (quelques centaines de Ko).
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .architecture import Architecture
from .config import Config
from .data import TEST_SEED, TRAIN_SEED, collect, collect_segments
from .training import CostTrainer, SelectionReadout, WorkspaceTrainer, WorldModelTrainer

TRAIN_MAPS, TEST_MAPS = 2000, 300


def train(out="checkpoints/step2.pt", verbose=True):
    torch.manual_seed(0)
    cfg = Config()
    arch = Architecture(cfg)
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)

    say(f"Collecte : {TRAIN_MAPS} cartes d'entraînement, {TEST_MAPS} cartes de test...")
    train_data = collect(cfg, n_maps=TRAIN_MAPS, seed=TRAIN_SEED)
    test_data = collect(cfg, n_maps=TEST_MAPS, seed=TEST_SEED)
    segments = collect_segments(cfg, n_maps=TRAIN_MAPS, seed=TRAIN_SEED)
    say(f"  {len(train_data)} transitions et {len(segments)} segments de 3 pas "
        f"d'entraînement, {len(test_data)} transitions de test\n")

    say("1/3 Modèle du monde (JEPA + dynamique inverse + événements, 1 et 3 pas)")
    wm_log = WorldModelTrainer(arch.world_model).fit(train_data, test_data, segments,
                                                     verbose=verbose)
    for p in arch.world_model.parameters():             # gelé pour la suite
        p.requires_grad = False

    say("\n2/3 Module de coût (sur les latents du modèle du monde)")
    cost_log = CostTrainer(arch.cost, arch.world_model).fit(train_data, test_data, verbose=verbose)

    say("\n3/3 Espace de travail (attention sélective sous goulot)")
    readout = SelectionReadout(cfg)
    ws_log = WorkspaceTrainer(arch.workspace, readout, arch.world_model).fit(
        train_data.next_obs, test_data.next_obs, verbose=verbose)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "architecture": arch.state_dict(),
                "readout": readout.state_dict(),
                "log": {"world_model": wm_log, "cost": cost_log, "workspace": ws_log}}, out)
    say(f"\nTerminé en {time.time() - t0:.0f} s. Sauvegardé dans {out}")
    return arch, readout, {"world_model": wm_log, "cost": cost_log, "workspace": ws_log}


def load(path="checkpoints/step2.pt"):
    ckpt = torch.load(path, weights_only=False)
    cfg = Config(**ckpt["config"])
    arch = Architecture(cfg)
    arch.load_state_dict(ckpt["architecture"])
    arch.eval()
    readout = SelectionReadout(cfg)
    readout.load_state_dict(ckpt["readout"])
    readout.eval()
    return arch, readout, ckpt["log"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m mini_iag.train")
    parser.add_argument("--out", default="checkpoints/step2.pt")
    args = parser.parse_args()
    log = train(args.out)[2]
    Path(args.out).with_suffix(".json").write_text(json.dumps(log, indent=1))
