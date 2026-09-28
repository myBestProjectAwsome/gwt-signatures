"""Entraîner le modèle des conséquences :  python -m mini_iag.train_effects   (~5 minutes)

Même expérience que tout le reste (le marcheur au hasard, cartes publiques).
Écrit checkpoints/effect_model.pt.
"""
import time
from pathlib import Path

import torch

from .config import Config
from .data import TRAIN_SEED
from .data_keydoor import collect_kd
from .modules.effect_model import EffectModel
from .training.effect_trainer import EffectTrainer

EFFECT_PATH = Path("checkpoints/effect_model.pt")


def train_effects(out=EFFECT_PATH, iters=6000, verbose=True):
    torch.manual_seed(0)
    t0 = time.time()
    data = collect_kd(Config.keydoor(), n_maps=3000, seed=TRAIN_SEED)
    model = EffectModel()
    log = EffectTrainer(model, iters=iters).fit(data, verbose=verbose)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "log": log}, out)
    if verbose:
        print(f"Terminé en {time.time() - t0:.0f} s. Sauvegardé dans {out}")
    return model


def load_effects(path=EFFECT_PATH):
    model = EffectModel()
    model.load_state_dict(torch.load(path, weights_only=False)["model"])
    model.eval()
    return model


if __name__ == "__main__":
    train_effects()
