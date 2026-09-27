"""Entraîner la carte mentale :  python -m mini_iag.train_map   (~25 minutes)

Q-learning hors ligne sur la même expérience que l'étape 4a (l'agent qui
marchait au hasard sur les cartes publiques). Écrit checkpoints/mental_map.pt.
Utilisation : load_task_agent(mental_map=MAP_PATH).

Version 2 (par défaut) : en plus de Bellman, l'auto-supervision des cases
(cell_weight=1) : chaque pas vécu renseigne la case où l'agent est arrivé.
La version 1 (cell_weight=0) est gardée pour comparaison (diagnostic 12) :
python -c "from mini_iag.train_map import *; train_map(MAP_V1_PATH, cell_weight=0)"
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
MAP_V1_PATH = Path("checkpoints/mental_map_v1.pt")      # version 1 (Bellman seul), pour comparaison
PARAMS = {"iterations": 25, "discount": 0.95}


def train_map(out=MAP_PATH, iters=12000, verbose=True, cell_weight=1.0, **params):
    torch.manual_seed(0)
    cfg = Config.keydoor()
    params = {**PARAMS, **params}
    t0 = time.time()
    data = collect_kd(cfg, n_maps=3000, seed=TRAIN_SEED)
    mental_map = MentalMap(cfg, **params)
    log = MapTrainer(mental_map, iters=iters, cell_weight=cell_weight).fit(data, verbose=verbose)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"map": mental_map.state_dict(), "params": params, "cell_weight": cell_weight,
                "log": log}, out)
    if verbose:
        print(f"Terminé en {time.time() - t0:.0f} s. Sauvegardé dans {out}")
    return mental_map


if __name__ == "__main__":
    train_map()
