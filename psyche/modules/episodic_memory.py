from collections import deque

from .base import Module
from ..proposal import Proposal


class EpisodicMemory(Module):
    """Mémoire épisodique : premier module qui UTILISE le contenu diffusé.

    À chaque broadcast :
      1. ENCODAGE : le contenu conscient est stocké, avec une importance égale
         à sa saillance (un pic CPU ou un message utilisateur marque plus
         qu'une charge de fond).
      2. RAPPEL ASSOCIATIF : le contenu sert d'indice (cue) ; on cherche
         l'épisode passé le plus proche (similarité x importance). La saillance
         du rappel est PROPORTIONNELLE à celle de l'indice : un contenu banal
         n'évoque qu'une réminiscence faible, un événement marquant (message,
         pic) ravive un souvenir fort.
      3. RAPPEL DÉLIBÉRÉ : si le contenu est "revoir un souvenir" (intention de
         l'explorateur), on propose le souvenir le plus important et le moins
         souvent rappelé. Deux modules se coordonnent via le workspace.

    Quand un souvenir gagne, il est renforcé (reconsolidation) et sert
    lui-même d'indice. Comme chaque rappel est plus faible que son indice,
    les chaînes d'associations s'éteignent d'elles-mêmes en 2 ou 3 sauts.

    Les phrases d'enveloppe ("cela me rappelle", "je me souviens") sont
    écrites à la main ; le CONTENU rappelé, lui, n'est jamais scripté.
    """
    name = "memoire"
    MIN_SIMILARITY = 0.4      # en dessous : pas d'association
    MIN_AGE = 5               # ignore les épisodes vécus dans les 5 derniers épisodes
    CUE_GAIN = 0.6            # saillance = gain x similarité x importance x saillance de l'indice
    DELIBERATE_SALIENCE = 0.6
    RECENT_RECALLS = 8        # un souvenir rappelé n'est pas re-proposé tout de suite

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.pending = None               # (Proposal, episode_id)
        self.recent = deque(maxlen=self.RECENT_RECALLS)
        self._offered_id = None

    # ------------------------------------------------------------ proposer
    def propose(self, h):
        if self.pending is None:
            return None
        proposal, self._offered_id = self.pending
        self.pending = None
        return proposal

    # ------------------------------------------------------------ recevoir
    def receive(self, broadcast, h):
        super().receive(broadcast, h)
        self.pending = None

        if broadcast.source == self.name:          # un de mes souvenirs a gagné
            eid = self._offered_id
            self.store.mark_recalled(eid)
            self.recent.append(eid)
            self._prepare_cue(self.store.get(eid).content, broadcast.salience)
            return

        self.store.encode(broadcast.content, broadcast.source, broadcast.salience)
        if broadcast.source == "exploration" and broadcast.content == "revoir un souvenir":
            self._prepare_deliberate()
        else:
            self._prepare_cue(broadcast.content, broadcast.salience)

    # ------------------------------------------------------------ rappels
    def _candidates(self, exclude_content=None):
        for eid, sim in self.store.similarities(exclude_content or ""):
            ep = self.store.get(eid)
            if ep.content == exclude_content or eid in self.recent:
                continue
            if self.store.clock - ep.last_episode < self.MIN_AGE:
                continue
            yield ep, sim

    def _prepare_cue(self, cue, cue_salience):
        best, best_score = None, 0.0
        for ep, sim in self._candidates(exclude_content=cue):
            if sim < self.MIN_SIMILARITY:
                continue
            score = sim * ep.importance
            if score > best_score:
                best, best_score = ep, score
        if best is None:
            return
        sal = self.CUE_GAIN * best_score * cue_salience
        self.pending = (Proposal(self.name, f"cela me rappelle : {best.content}", sal),
                        best.id)

    def _prepare_deliberate(self):
        cands = [ep for ep, _ in self._candidates()]
        if not cands:
            return
        best = max(cands, key=lambda ep: (ep.importance / (1 + ep.recalls), -ep.id))
        self.pending = (Proposal(self.name, f"je me souviens : {best.content}",
                                 self.DELIBERATE_SALIENCE), best.id)
