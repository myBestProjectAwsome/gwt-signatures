"""Boucle principale : percevoir -> proposer -> compétition -> broadcast -> homéostasie.

Sécurité : aucun module n'a accès aux signaux/processus. Ctrl+C arrête tout.
La mémoire n'écrit que dans son fichier SQLite.
"""
import time

from .environment import Environment
from .hormones import Hormones
from .memory_store import MemoryStore
from .workspace import GlobalWorkspace
from .modules import (SystemSensor, Explorer, Social, Metacognition, UserInput,
                      EpisodicMemory)


def build(env, store=None, memory=True):
    """Assemble psyche. `store` : MemoryStore (par défaut en RAM, non persistant).
    `memory=False` : version sans mémoire (utile pour les ablations)."""
    h = Hormones()
    ws = GlobalWorkspace()
    user = UserInput()
    modules = [SystemSensor(env), Explorer(), Social(), Metacognition(ws), user]
    if memory:
        modules.append(EpisodicMemory(store if store is not None else MemoryStore()))
    return h, ws, user, modules


def step(env, h, ws, user, modules):
    """Un cycle complet. Renvoie (gagnant ou None, propositions du cycle)."""
    msg = env.poll_user()
    if msg:
        h.on_user_message()
        user.hear(msg)
    proposals = [p for p in (m.propose(h) for m in modules) if p is not None]
    winner = ws.compete(proposals)
    if winner:
        for m in modules:              # réentrance : broadcast à tous
            m.receive(winner, h)
    h.update(ws, env)
    return winner, proposals


def run(cycles=None, period=1.0, env=None, db_path="data/episodes.sqlite", memory=True):
    env = env or Environment()
    store = MemoryStore(db_path) if memory else None
    h, ws, user, modules = build(env, store, memory)
    print("psyche démarre. Tape un message + Entrée pour lui parler, Ctrl+C pour arrêter.")
    if store is not None:
        print(f"mémoire : {db_path} ({len(store)} souvenirs, {store.clock} épisodes vécus)")
    t = 0
    try:
        while cycles is None or t < cycles:
            winner, _ = step(env, h, ws, user, modules)
            if winner:
                print(f"[{t:04d}] {h} | {winner.source:>15} : {winner.content}")
            t += 1
            time.sleep(period / max(h.energie, 0.2))  # moins d'énergie = plus lent
    except KeyboardInterrupt:
        print("\nArrêt demandé. Au revoir.")
    finally:
        if store is not None:
            store.close()
