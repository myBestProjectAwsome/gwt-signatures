"""Planification par événements (feuille de route v2, lot 3) : découvrir les prérequis.

On ne donne à l'agent que le but final (ex. l'objectif). Avant d'agir, il se
demande : « faut-il d'abord provoquer un autre événement ? ». Pour cela il
imagine des suites courtes d'événements qui finissent par le but :
    [objectif]      [clé, objectif]      [porte, objectif]      [clé, porte, objectif] …
et estime le coût de chacune avec ses propres modèles appris :
  - la CARTE MENTALE donne la distance jusqu'au prochain événement (ou « inatteignable ») ;
  - le MODÈLE DES CONSÉQUENCES imagine le monde APRÈS cet événement ;
  - la carte mentale, recalculée sur ce monde imaginé, dit ce qui est devenu atteignable.
Il retient la suite la plus courte, et vise son premier événement (le sous-but).

Aucune règle « si porte, alors clé » n'est écrite : si la clé est utile, c'est
parce que le modèle des conséquences a appris qu'après la clé le monde change,
et que la carte mentale, sur ce monde changé, trouve un chemin. Si un détour
sans clé est plus court, la suite [objectif] gagne ; si le passage est déjà
ouvert, aussi.

Ce qui est fourni (déclaré dans CONNAISSANCES_FOURNIES.md) : la liste des
événements possibles, la profondeur de recherche (3 événements), et l'idée
qu'un événement peut changer le monde.
"""
import math

import torch

from ..modules.mental_map import TARGETS


class EventPlanner:
    def __init__(self, mental_map, effect_model, max_depth=3):
        self.map, self.effects = mental_map, effect_model
        self.max_depth = max_depth
        self.log_gamma = math.log(mental_map.discount)

    @torch.no_grad()
    def reach(self, obs, blocked=None):
        """Pour chaque événement : (distance estimée, case) ou None si inatteignable.
        blocked : fonction (observation) -> cases constatées bloquantes dans cette situation."""
        b = blocked(obs[0]) if blocked is not None else None
        q = self.map(obs, b)[0]                                # (3, 4)
        U = self.map.values(obs, b)[0]                         # (3, H, W)
        r, _, _ = self.map.maps(obs, b)
        out = {}
        for k, e in enumerate(TARGETS):
            best = float(q[k].max())
            cand = (r[0, k] > 0.5).float() * U[k]
            if best < 1e-3 or float(cand.max()) <= 0:
                out[e] = None
                continue
            cell = divmod(int(cand.flatten().argmax()), cand.shape[-1])
            out[e] = (1 + math.log(best) / self.log_gamma, cell)
        return out

    @torch.no_grad()
    def plan(self, obs, final, blocked=None):
        """Meilleure suite d'événements finissant par `final` : (suite, coût) ou (None, inf)."""
        obs = torch.as_tensor(obs)[None].float()
        best = [None, float("inf")]

        def search(o, seq, cost, depth):
            reach = self.reach(o, blocked)
            if reach[final] is not None:
                total = cost + reach[final][0] + 0.01 * len(seq)
                if total < best[1]:
                    best[:] = [seq + [final], total]
            if depth + 1 >= self.max_depth:
                return
            for e in TARGETS:
                if e == final or (seq and seq[-1] == e) or reach[e] is None:
                    continue
                d, cell = reach[e]
                if cost + d >= best[1]:
                    continue
                mask = torch.zeros(o.shape[-2:])
                mask[cell] = 1
                search(self.effects.imagine(o, mask[None]), seq + [e], cost + d, depth + 1)

        search(obs, [], 0.0, 0)
        return best[0], best[1]
