"""Encodeur : transforme un texte en vecteur pour mesurer des similarités.

Choix volontairement simple et sans dépendance (pas de modèle à télécharger) :
"feature hashing" de mots et de trigrammes de caractères.

  "CPU à 5%"  et  "CPU à 7%"          -> très proches (mêmes trigrammes)
  "l'utilisateur dit : bonjour"
  et "l'utilisateur dit : bonjour psyche" -> proches
  "observer l'heure" et "CPU à 5%"   -> éloignés

Limite assumée : c'est une similarité de SURFACE (orthographe), pas de sens.
"salut" et "bonjour" sont éloignés. Un encodeur sémantique (sentence-transformers)
pourra remplacer celui-ci sans rien changer au reste : même interface encode().

Déterminisme : on n'utilise PAS hash() de Python, randomisé à chaque lancement
(cf. PYTHONHASHSEED), mais zlib.crc32, identique sur toutes les machines.
"""
import re
import unicodedata
import zlib

import numpy as np


class HashingEncoder:
    def __init__(self, dim=512):
        self.dim = dim

    @staticmethod
    def _normalize(text):
        text = unicodedata.normalize("NFKD", text.lower())
        return "".join(c for c in text if not unicodedata.combining(c))

    def _features(self, text):
        words = re.findall(r"[a-z0-9]+", self._normalize(text))
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def encode(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for f in self._features(text):
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0  # signe : limite les collisions
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
