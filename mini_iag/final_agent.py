"""L'agent présenté au test final (étape 5), et sa référence « sans connaissances ».

Agent final : l'agent de l'étape 4a (modèle du monde, coût configurable,
critique d'actions) + la carte mentale, sans vie préalable (pour que le
résultat soit reproductible par n'importe qui à partir des mêmes commandes).
Ses VALEURS sont verrouillées : lock() gèle le module de coût et enregistre
son empreinte SHA-256, qui couvre aussi w_danger et w_succès (règle V4).

Référence sans connaissances (règle V2) : la MÊME architecture (mêmes modules,
même carte mentale, mêmes réglages), avec des poids ALÉATOIRES. Elle suit la
même procédure d'adaptation (practice, 50 épisodes par tâche).

Témoins de la feuille de route v2 (lot 1) :
  build_frozen_copy()   l'agent final, sans aucune mise à jour pendant l'adaptation :
                        isole l'apport de l'adaptation elle-même ;
  ScratchLearner        (scratch_learner.py) une architecture repartant de zéro dont
                        TOUS les modules sont entraînables sur l'expérience d'adaptation :
                        mesure le bénéfice des acquis antérieurs.
build_relearner() est gardé pour l'historique : c'est la référence du premier test,
trop faible parce qu'une partie de ses modules aléatoires n'apprenait jamais.

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


def build_frozen_copy(step4=STEP4, mental_map=MAP_PATH):
    """Copie figée : même agent, mais practice() ne change rien."""
    agent = build_final_agent(step4, mental_map)
    agent.practice = lambda *args, **kwargs: None
    agent.frozen = True
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


# ---------------------------------------------------------------- lot 3 : découvrir les prérequis
class NoEffects:
    """Ablation : un « modèle des conséquences » qui croit que rien ne change jamais."""

    def imagine(self, obs, cell_mask):
        return obs


def build_prerequisite_agent(step4=STEP4, mental_map=MAP_PATH, effects="checkpoints/effect_model.pt",
                             ablate_effects=False, **flags):
    """L'agent du verdict + la planification par événements (il ne reçoit que le but final)."""
    from .prerequisite_agent import PrerequisiteAgent
    from .train_effects import load_effects
    base = load_task_agent(step4, mental_map=mental_map, **flags)
    effect = NoEffects() if ablate_effects else load_effects(effects)
    agent = PrerequisiteAgent(base.arch, base.cfg, base.core.cue, base.core.sigma, effect, **flags)
    agent.cost_module.lock()
    return agent


def _verdict_weights():
    from .reproduire import VERDICT
    return VERDICT / "step4.pt", VERDICT / "mental_map.pt"


from .train_map import MAP_V3_PATH as MAP_V3   # noqa: E402  (carte v3 : cases jugées sans le canal de l'agent)

EXTRA_AGENTS = {
    "carte v3 seule": lambda: load_task_agent(_verdict_weights()[0], mental_map=MAP_V3,
                                              cfg_overrides={"map_weight": 1000.0}),
    "prérequis (carte v3 + conséquences)": lambda: build_prerequisite_agent(_verdict_weights()[0], MAP_V3),
    "prérequis (carte v3), sans modèle des conséquences": lambda: build_prerequisite_agent(
        _verdict_weights()[0], MAP_V3, ablate_effects=True),
}
