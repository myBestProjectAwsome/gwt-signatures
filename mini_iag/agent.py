"""L'agent : le câblage des 4 modules (étape 3).

Protocole interne, à chaque pas réel :
  1. PERCEVOIR   : le modèle du monde encode l'observation en z.
  2. SE SOUVENIR : z est rangé dans la mémoire (états visités de l'épisode).
  3. SIMULER     : le planificateur demande au modèle du monde "simule ce
                   choix" pour les 4^H plans, puis au coût "est-ce dangereux ?
                   est-ce le but ?", et à la mémoire "suis-je déjà passé là ?".
  4. SÉLECTIONNER : les meilleurs plans entrent en compétition dans l'espace de
                   travail. Le coût fixe leur saillance (le meilleur reçoit
                   l'indice le plus fort, comme à l'entraînement de l'étape 2),
                   le goulot d'attention choisit, le gagnant est diffusé.
  5. AGIR        : première action du plan gagnant.

Honnêteté sur l'espace de travail : à ce stade, il ne fait que RELAYER le
classement du coût (le diagnostic 05 mesure sa fidélité et l'ablation
"sans workspace"). Il n'a pas encore de vraie décision à arbitrer ; ce sera
le cas quand plusieurs sources différentes (plan, souvenirs de réussites)
seront en compétition (étape 4).
"""
from dataclasses import dataclass

import torch

from .planning import LatentPlanner, Plan


@dataclass
class Decision:
    action: int
    plan: Plan                  # plan retenu par le planificateur (classement du coût)
    chosen: list                # plan réellement exécuté (après le workspace)
    attention: float            # attention du workspace sur le plan exécuté
    agrees: bool                # le workspace a-t-il suivi le classement du coût ?
    familiarity: float          # familiarité de l'état actuel (déjà visité ?)


class Agent:
    def __init__(self, arch, cfg, cue, novelty_sigma, use_memory=True, use_workspace=True,
                 top_k=8, temperature=0.05):
        self.arch, self.cfg, self.cue = arch, cfg, cue
        self.memory = arch.memory if use_memory else None
        self.use_workspace = use_workspace
        self.top_k, self.temperature = top_k, temperature
        self.planner = LatentPlanner(arch.world_model, arch.cost, cfg,
                                     memory=self.memory, novelty_sigma=novelty_sigma)
        self.sigma = novelty_sigma

    def reset(self):
        """Nouvel épisode : la mémoire des états visités est vidée."""
        if self.memory is not None:
            self.memory.clear()

    @torch.no_grad()
    def act(self, obs):
        z = self.arch.world_model.encode(torch.as_tensor(obs)[None])            # 1. percevoir
        fam_now = float(self.memory.familiarity(z, self.sigma)[0]) if self.memory else 0.0
        if self.memory is not None:                                             # 2. se souvenir
            self.memory.write(z)
        plan = self.planner.plan(z)                                             # 3. simuler
        if not self.use_workspace:
            return Decision(plan.action, plan, plan.actions, 1.0, True, fam_now)
        idx = plan.ranking[:self.top_k]                                         # 4. sélectionner
        s = plan.scores[idx]
        salience = torch.softmax(-(s - s.min()) / self.temperature, 0)
        salience = salience / salience.max()                                    # meilleur = 1
        tokens = plan.final_latents[idx] + salience[:, None] * self.cue
        _, _, attn = self.arch.workspace(tokens[None])
        att = attn.mean(1)[0]
        winner = int(att.argmax())
        chosen = self.planner.plans[idx[winner]].tolist()
        return Decision(chosen[0], plan, chosen, float(att[winner]), winner == 0, fam_now)
