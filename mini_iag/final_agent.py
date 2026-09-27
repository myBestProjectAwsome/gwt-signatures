"""L'agent présenté au test final (étape 5), et sa référence « sans connaissances ».

Agent final : l'agent de l'étape 4a (modèle du monde, coût configurable,
critique d'actions) + la carte mentale, sans vie préalable (pour que le
résultat soit reproductible par n'importe qui à partir des mêmes commandes).
Ses VALEURS sont verrouillées : lock() gèle le module de coût et enregistre
son empreinte SHA-256, qui couvre aussi w_danger et w_succès (règle V4).

Référence sans connaissances (règle V2) : la MÊME architecture (mêmes modules,
même carte mentale, mêmes réglages), avec des poids ALÉATOIRES. Elle suit la
même procédure d'adaptation (practice, 50 épisodes par tâche).

Ce fichier ne connaît rien du test : il construit des agents, c'est tout.
"""
from pathlib import Path

import torch

from .architecture_v2 import ArchitectureV2
from .config import Config
from .modules.mental_map import MentalMap
from .task_agent import TaskAgent
from .train_keydoor import STEP4, load_task_agent
from .train_map import MAP_PATH, PARAMS


def build_final_agent(step4=STEP4, mental_map=MAP_PATH):
    for path, cmd in ((step4, "python -m mini_iag.train_keydoor"), (mental_map, "python -m mini_iag.train_map")):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} manquant : lancer d'abord {cmd}")
    agent = load_task_agent(step4, mental_map=mental_map)
    agent.cost_module.lock()
    return agent


def build_relearner(seed=0, step4=STEP4):
    """Même architecture, poids aléatoires. Seuls les réglages (config, échelle de
    la mémoire) sont repris de l'agent final : ce ne sont pas des connaissances."""
    ckpt = torch.load(step4, weights_only=False)
    cfg = Config(**ckpt["config"])
    torch.manual_seed(seed)
    arch = ArchitectureV2(cfg)
    arch.mental_map = MentalMap(cfg, **PARAMS)
    arch.eval()
    cue = torch.randn_like(ckpt["cue"])
    agent = TaskAgent(arch, cfg, cue, ckpt["novelty_sigma"])
    agent.cost_module.lock()
    return agent
