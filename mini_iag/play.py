"""Regarder l'agent jouer.  Lancer :  python -m mini_iag.play [--carte N]

Affiche la carte à chaque pas, avec ce que l'agent "pense" : le plan qu'il a
imaginé, le danger et le succès qu'il prévoit, s'il reconnaît l'endroit, et
l'attention de son espace de travail. Les cartes de démonstration ne font pas
partie des cartes d'entraînement.
"""
import argparse
import sys
import time

from .coordinate import STEP3, coordinate, load_agent
from .environment import GridWorld
from .environment.gridworld import ACTION_NAMES

ARROWS = ["↑", "↓", "←", "→"]
DEMO_SEED = 90_000


def show(env, dec, t, clear):
    if clear:
        print("\033[2J\033[H", end="")
    plan = " ".join(ARROWS[a] for a in dec.chosen)
    p = dec.plan
    print(f"pas {t}\n")
    print(env.render())
    print(f"\n  plan imaginé      {plan}   ({p.n_plans} plans simulés)")
    print("  danger prévu      " + " ".join(f"{d:.2f}" for d in p.danger))
    print("  succès prévu      " + " ".join(f"{s:.2f}" for s in p.success))
    print(f"  déjà vu ici ?     {dec.familiarity:.2f}   (0 = nouveau)")
    print(f"  workspace         attention {dec.attention:.2f} sur le plan retenu"
          + ("" if dec.agrees else "   (a préféré un autre plan que le coût)"))
    print(f"  action            {ARROWS[dec.action]} {ACTION_NAMES[dec.action]}")


def main():
    parser = argparse.ArgumentParser(prog="python -m mini_iag.play")
    parser.add_argument("--carte", type=int, default=0, help="numéro de carte de démo")
    parser.add_argument("--delai", type=float, default=0.6, help="secondes entre deux pas")
    parser.add_argument("--sans-memoire", action="store_true")
    args = parser.parse_args()
    if not STEP3.exists():
        print("Pas de checkpoint de l'étape 3 : câblage (python -m mini_iag.coordinate)...")
        coordinate(verbose=False)
    agent = load_agent(use_memory=not args.sans_memoire)
    env = GridWorld(agent.cfg)
    obs = env.reset(seed=DEMO_SEED + args.carte)
    optimal = env.distance_to_goal()
    agent.reset()
    clear = sys.stdout.isatty()
    for t in range(1, agent.cfg.max_steps + 1):
        dec = agent.act(obs)
        show(env, dec, t, clear)
        obs, _, done, info = env.step(dec.action)
        time.sleep(args.delai)
        if done:
            break
    if clear:
        print("\033[2J\033[H", end="")
    print(env.render())
    if info["success"]:
        print(f"\nObjectif atteint en {t} pas (chemin le plus court : {optimal}).")
    elif info["danger"]:
        print(f"\nTombé dans la lave au pas {t}.")
    else:
        print(f"\nBloqué : objectif non atteint en {t} pas (chemin le plus court : {optimal}).")


if __name__ == "__main__":
    main()
