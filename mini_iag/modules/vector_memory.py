"""Mémoire vectorielle : stocke des vecteurs latents et leurs conséquences.

Pas de poids à apprendre : c'est une base de données dans l'espace latent.
Chaque entrée associe une clé (l'état latent) à ce qui s'est passé ensuite
(action, récompense, danger, succès). Rappel par similarité cosinus.

Quand la capacité est atteinte, les plus anciennes entrées sont écrasées
(tampon circulaire).
"""
import torch
from torch.nn import functional as F


class VectorMemory:
    FIELDS = ("action", "reward", "danger", "success")

    def __init__(self, cfg):
        self.capacity = cfg.memory_capacity
        self.keys = torch.zeros(self.capacity, cfg.latent_dim)
        self.values = {f: torch.zeros(self.capacity) for f in self.FIELDS}
        self.size = 0
        self._next = 0

    def __len__(self):
        return self.size

    @torch.no_grad()
    def write(self, z, **values):
        """z : (B, D) ; values : champs de FIELDS, chacun de forme (B,)."""
        for i in range(z.shape[0]):
            self.keys[self._next] = z[i]
            for f in self.FIELDS:
                self.values[f][self._next] = float(values.get(f, torch.zeros(z.shape[0]))[i])
            self._next = (self._next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    @torch.no_grad()
    def recall(self, query, k=5):
        """query : (B, D). Renvoie (similarités (B, k), indices (B, k))."""
        if self.size == 0:
            raise ValueError("mémoire vide")
        keys = F.normalize(self.keys[:self.size], dim=-1)
        sims = F.normalize(query, dim=-1) @ keys.T
        return sims.topk(min(k, self.size), dim=-1)

    def value(self, field, indices):
        return self.values[field][indices]

    def clear(self):
        self.size = 0
        self._next = 0

    @torch.no_grad()
    def familiarity(self, queries, sigma):
        """"Suis-je déjà passé par là ?" pour chaque requête (M, D).

        Somme de noyaux gaussiens exp(-||q - k||² / sigma²) sur les clés stockées :
        ~0 pour un état jamais vu, ~1 par visite d'un état identique.
        Distance euclidienne plutôt que cosinus : dans le latent, la carte domine
        la direction du vecteur, et tous les états d'une même carte auraient un
        cosinus proche de 1.
        """
        if self.size == 0:
            return torch.zeros(len(queries))
        d2 = torch.cdist(queries, self.keys[:self.size]) ** 2
        return torch.exp(-d2 / sigma ** 2).sum(-1)
