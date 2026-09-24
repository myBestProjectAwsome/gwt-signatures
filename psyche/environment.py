"""Environnement : tout ce que psyche perçoit du monde extérieur.

Les modules ne lisent JAMAIS psutil ou stdin directement : ils passent par
un objet Environment. On peut ainsi remplacer le vrai monde par un monde
simulé (FakeEnvironment) pour obtenir des tests reproductibles, identiques
sur toutes les machines.
"""
import random
import sys

try:
    import psutil
except ImportError:
    psutil = None


class Environment:
    """Le vrai monde : capteurs psutil + messages tapés dans le terminal."""

    def battery(self):
        """Niveau de batterie entre 0 et 1, ou None si pas de batterie."""
        batt = psutil.sensors_battery() if psutil else None
        return None if batt is None else batt.percent / 100

    def cpu(self):
        """Charge CPU entre 0 et 1."""
        return psutil.cpu_percent(interval=None) / 100 if psutil else 0.0

    def poll_user(self):
        """Message tapé par l'utilisateur (Entrée pour valider), sans bloquer.
        Fonctionne sous Linux/macOS ; renvoie toujours None sous Windows."""
        try:
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                line = sys.stdin.readline().strip()
                return line or None
        except (ImportError, OSError, ValueError):
            pass
        return None


class FakeEnvironment:
    """Monde simulé et reproductible, pour les diagnostics.

    - batterie : se vide de 100 % à 20 %, puis se recharge, en boucle
    - CPU      : charge de fond aléatoire (graine fixée) avec des pics rares
    - utilisateur : un message en moyenne tous les `user_period` cycles
    """

    def __init__(self, seed=0, drain_cycles=400, user_period=60):
        self.rng = random.Random(seed)
        self.t = 0
        self.drain_cycles = drain_cycles
        self.user_period = user_period

    def tick(self):
        self.t += 1

    def battery(self):
        phase = self.t % (2 * self.drain_cycles)
        if phase < self.drain_cycles:                      # décharge
            return 1.0 - 0.8 * phase / self.drain_cycles
        return 0.2 + 0.8 * (phase - self.drain_cycles) / self.drain_cycles  # charge

    def cpu(self):
        if self.rng.random() < 0.05:
            return self.rng.uniform(0.6, 0.95)                 # pic de charge
        return self.rng.uniform(0.02, 0.2)

    def poll_user(self):
        if self.user_period and self.rng.random() < 1 / self.user_period:
            return "bonjour"
        return None
