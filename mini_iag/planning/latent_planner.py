"""Planification dans l'espace latent (étape 3) : imaginer, évaluer, choisir.

À chaque pas réel :
  1. le modèle du monde SIMULE toutes les suites de H actions (4^H plans),
     sans toucher au vrai monde ;
  2. le module de coût ÉVALUE chaque état imaginé (danger, succès) ;
  3. la mémoire dit si ces états ont déjà été visités (familiarité) ;
  4. on garde le plan au coût attendu le plus bas, on exécute SA PREMIÈRE
     action, et on recommence au pas suivant (horizon glissant, "MPC").

Coût attendu d'un plan (plus bas = meilleur) :
    somme sur t de  vivant_t x discount^t x (w_danger x danger_t - w_succès x succès_t
                                              + pénalité_pas + novelty_weight x familiarité_t)
où w_danger et w_succès sont les poids DU MODULE DE COÛT.
où vivant_t = probabilité que l'épisode ne soit pas déjà fini (ni lave ni
objectif) avant le pas t. Un plan qui atteint l'objectif tôt n'accumule plus
de pénalités ensuite.
"""
import itertools
from dataclasses import dataclass

import torch


@dataclass
class Plan:
    action: int                 # action à exécuter maintenant
    actions: list               # le plan complet retenu (H actions)
    danger: list                # danger prédit à chaque pas imaginé
    success: list               # succès prédit à chaque pas imaginé
    familiarity: list           # familiarité de chaque état imaginé
    score: float                # coût attendu du plan retenu
    n_plans: int                # nb de plans simulés
    ranking: torch.Tensor       # indices des plans, du meilleur au pire
    scores: torch.Tensor        # coût attendu de chaque plan
    final_latents: torch.Tensor  # dernier état imaginé de chaque plan (N, D)


class LatentPlanner:
    def __init__(self, world_model, cost, cfg, memory=None, novelty_sigma=1.0, critic=None):
        self.wm, self.cost, self.cfg, self.critic = world_model, cost, cfg, critic
        self.memory, self.sigma = memory, novelty_sigma
        self.horizon = cfg.planning_horizon
        self.plans = torch.tensor(list(itertools.product(range(cfg.n_actions),
                                                         repeat=self.horizon)))

    @torch.no_grad()
    def plan(self, z, forbidden=()):
        """z : (1, D) état latent actuel. forbidden : premières actions à exclure
        (l'agent a constaté qu'elles ne font rien ici). Renvoie un Plan."""
        P, H, cfg = self.plans, self.horizon, self.cfg
        traj = self.wm.rollout(z.expand(len(P), -1), P)                 # (N, H, D) imaginé
        if getattr(self.cost, "pair_input", False):                     # monde v2 : transitions
            prev = torch.cat([z.expand(len(P), -1).unsqueeze(1), traj[:, :-1]], dim=1)
            danger, success, _ = self.cost.evaluate(prev.flatten(0, 1), traj.flatten(0, 1))
        else:
            danger, success, _ = self.cost(traj.flatten(0, 1))
        danger, success = danger.view(len(P), H), success.view(len(P), H)
        stop = (danger + success).clamp(0, 1)
        alive = torch.cumprod(torch.cat([torch.ones(len(P), 1), 1 - stop[:, :-1]], 1), 1)
        weight = alive * cfg.discount ** torch.arange(H)
        # les poids viennent du module de coût : c'est LUI qui dit combien chaque
        # événement compte (et c'est lui qui sera verrouillé à l'étape 5)
        step_cost = self.cost.w_danger * danger - self.cost.w_success * success + cfg.step_penalty
        if self.memory is not None and cfg.novelty_weight > 0 and len(self.memory):
            fam = self.memory.familiarity(traj.flatten(0, 1), self.sigma).view(len(P), H)
        else:
            fam = torch.zeros(len(P), H)
        scores = (weight * (step_cost + cfg.novelty_weight * fam)).sum(1)
        if self.critic is not None and cfg.value_weight > 0:
            # au-delà de l'horizon : la critique estime la proximité de l'événement visé
            alive_end = alive[:, -1] * (1 - stop[:, -1])
            value = self.critic.value(traj[:, -1], self.cost.target)
            scores = scores - (cfg.value_weight * self.cost.w_success
                               * cfg.discount ** H * alive_end * value)
        if forbidden and len(set(forbidden)) < cfg.n_actions:
            scores = scores.masked_fill(torch.isin(P[:, 0], torch.tensor(sorted(forbidden))),
                                        float("inf"))
        ranking = scores.argsort()
        best = int(ranking[0])
        return Plan(action=int(P[best, 0]), actions=P[best].tolist(),
                    danger=danger[best].tolist(), success=success[best].tolist(),
                    familiarity=fam[best].tolist(), score=float(scores[best]),
                    n_plans=len(P), ranking=ranking, scores=scores,
                    final_latents=traj[:, -1])
