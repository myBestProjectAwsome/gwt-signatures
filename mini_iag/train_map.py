"""Entraîner la carte mentale :  python -m mini_iag.train_map   (~25 minutes)

Q-learning hors ligne sur la même expérience que l'étape 4a (l'agent qui
marchait au hasard sur les cartes publiques). Écrit checkpoints/mental_map.pt.
Utilisation : load_task_agent(mental_map=MAP_PATH).
"""
import time
from pathlib import Path

import torch

from .config import Config
from .data import TRAIN_SEED
from .data_keydoor import collect_kd
from .modules.mental_map import MentalMap
from .training import MapTrainer

MAP_PATH = Path("checkpoints/mental_map.pt")
PARAMS = {"iterations": 25, "discount": 0.95}


def train_map(out=MAP_PATH, iters=12000, verbose=True, **params):
    torch.manual_seed(0)
    cfg = Config.keydoor()
    params = {**PARAMS, **params}
    t0 = time.time()
    data = collect_kd(cfg, n_maps=3000, seed=TRAIN_SEED)
    mental_map = MentalMap(cfg, **params)
    log = MapTrainer(mental_map, iters=iters).fit(data, verbose=verbose)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"map": mental_map.state_dict(), "params": params, "log": log}, out)
    if verbose:
        print(f"Terminé en {time.time() - t0:.0f} s. Sauvegardé dans {out}")
    return mental_map


if __name__ == "__main__":
    train_map()
