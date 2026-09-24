"""Boucle principale : proposer -> compétition -> broadcast -> homéostasie.

Sécurité : aucun module n'a accès aux signaux/processus. Ctrl+C arrête tout.
"""
import time

from .hormones import Hormones
from .workspace import GlobalWorkspace
from .modules import SystemSensor, Explorer, Social, Metacognition


def run(cycles=None, period=1.0):
    h = Hormones()
    ws = GlobalWorkspace()
    modules = [SystemSensor(), Explorer(), Social(), Metacognition(ws)]
    t = 0
    try:
        while cycles is None or t < cycles:
            proposals = [m.propose(h) for m in modules]
            winner = ws.compete(proposals)
            if winner:
                for m in modules:          # réentrance : broadcast à tous
                    m.receive(winner)
                print(f"[{t:04d}] {h} | {winner.source:>15} : {winner.content}")
            h.update(ws)
            t += 1
            time.sleep(period / max(h.energie, 0.2))  # moins d'énergie = plus lent
    except KeyboardInterrupt:
        print("\nArrêt demandé. Au revoir.")
