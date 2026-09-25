"""Stockage des épisodes dans SQLite (bibliothèque standard, un seul fichier).

Un épisode = un contenu qui a gagné le workspace (donc été "conscient").

Consolidation : un contenu déjà connu n'est pas dupliqué ; on incrémente
son compteur et on met à jour sa date. La base reste petite (quelques
centaines de lignes) même après des jours de fonctionnement.

Horloge épisodique : un compteur global, persistant d'une session à l'autre,
qui avance à chaque encodage. L'"âge" d'un souvenir se mesure dans cette
horloge (nombre d'épisodes vécus depuis), pas en secondes.
"""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoder import HashingEncoder

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY,
    content       TEXT UNIQUE NOT NULL,
    source        TEXT NOT NULL,
    vector        BLOB NOT NULL,
    importance    REAL NOT NULL,     -- saillance max au moment de l'encodage
    count         INTEGER NOT NULL,  -- nb de fois vécu
    recalls       INTEGER NOT NULL,  -- nb de fois rappelé (reconsolidation)
    first_episode INTEGER NOT NULL,  -- horloge épisodique
    last_episode  INTEGER NOT NULL,
    first_seen    REAL NOT NULL,     -- horodatage réel (information seulement)
    last_seen     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


@dataclass
class Episode:
    id: int
    content: str
    source: str
    importance: float
    count: int
    recalls: int
    first_episode: int
    last_episode: int


class MemoryStore:
    def __init__(self, path=":memory:", encoder=None):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self.encoder = encoder or HashingEncoder()
        row = self.db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
        self.clock = row[0] if row else 0
        self._load_index()

    # ------------------------------------------------------------ index RAM
    def _load_index(self):
        """Copie des vecteurs en mémoire vive pour des recherches rapides."""
        self._episodes, vecs = {}, []
        for r in self.db.execute(
                "SELECT id, content, source, importance, count, recalls, "
                "first_episode, last_episode, vector FROM episodes ORDER BY id"):
            self._episodes[r[0]] = Episode(*r[:8])
            vecs.append(np.frombuffer(r[8], dtype=np.float32))
        self._ids = list(self._episodes)
        self._matrix = np.vstack(vecs) if vecs else np.zeros((0, self.encoder.dim), np.float32)

    def __len__(self):
        return len(self._episodes)

    def get(self, episode_id):
        return self._episodes.get(episode_id)

    def episodes(self):
        return list(self._episodes.values())

    # ------------------------------------------------------------ écriture
    def encode(self, content, source, importance):
        """Enregistre un contenu conscient. Renvoie l'id de l'épisode."""
        self.clock += 1
        now = time.time()
        row = self.db.execute("SELECT id FROM episodes WHERE content=?", (content,)).fetchone()
        if row:                                   # consolidation
            eid = row[0]
            ep = self._episodes[eid]
            ep.count += 1
            ep.importance = max(ep.importance, importance)
            ep.last_episode = self.clock
            self.db.execute("UPDATE episodes SET count=?, importance=?, last_episode=?, "
                            "last_seen=? WHERE id=?",
                            (ep.count, ep.importance, self.clock, now, eid))
        else:                                     # nouvel épisode
            vec = self.encoder.encode(content)
            cur = self.db.execute(
                "INSERT INTO episodes (content, source, vector, importance, count, recalls, "
                "first_episode, last_episode, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?, ?)",
                (content, source, vec.tobytes(), importance, self.clock, self.clock, now, now))
            eid = cur.lastrowid
            self._episodes[eid] = Episode(eid, content, source, importance, 1, 0,
                                          self.clock, self.clock)
            self._ids.append(eid)
            self._matrix = np.vstack([self._matrix, vec[None, :]])
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('clock', ?)", (self.clock,))
        self.db.commit()
        return eid

    def mark_recalled(self, episode_id):
        """Reconsolidation : un souvenir rappelé est renforcé."""
        ep = self._episodes[episode_id]
        ep.recalls += 1
        self.db.execute("UPDATE episodes SET recalls=? WHERE id=?", (ep.recalls, episode_id))
        self.db.commit()

    # ------------------------------------------------------------ lecture
    def similarities(self, text):
        """Similarité cosinus entre `text` et chaque épisode : liste (id, sim)."""
        if not self._ids:
            return []
        sims = self._matrix @ self.encoder.encode(text)
        return list(zip(self._ids, sims.tolist()))

    def close(self):
        self.db.close()
