"""Boucle principale : percevoir -> proposer -> compétition -> broadcast -> homéostasie.

Sécurité : aucun module n'a accès aux signaux/processus. Ctrl+C arrête tout.
"""
import time

from .environment import Environment
from .hormones import Hormones
from .workspace import GlobalWorkspace
from .modules import SystemSensor, Explorer, Social, Metacognition, UserInput


def build(env):
    h = Hormones()
    ws = GlobalWorkspace()
    user = UserInput()
    modules = [SystemSensor(env), Explorer(), Social(), Metacognition(ws), user]
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


def run(cycles=None, period=1.0, env=None):
    env = env or Environment()
    h, ws, user, modules = build(env)
    print("psyche démarre. Tape un message + Entrée pour lui parler, Ctrl+C pour arrêter.")
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
