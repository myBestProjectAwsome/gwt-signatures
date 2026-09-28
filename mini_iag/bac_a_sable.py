"""Bac à sable : interagir avec la mini-IAG.   python -m mini_iag.bac_a_sable [--carte N]

Elle ne parle pas : elle ne connaît que sa grille et trois événements (clé,
porte, objectif). On communique donc avec elle par les canaux de l'architecture
de LeCun :
  - le CONFIGURATEUR : tu lui donnes un but (une suite d'événements), même un
    but qu'elle n'a jamais reçu. C'est ta façon de lui poser une question ;
  - le MONDE : tu modifies sa carte, même pendant qu'elle agit ;
  - le MODÈLE DU MONDE : tu lui demandes « que se passe-t-il si tu fais ça ? »,
    et elle répond avec son imagination (sans bouger) ;
  - la CARTE MENTALE et le MODULE DE COÛT : tu lis ce qu'elle trouve prometteur
    ou dangereux, case par case.

Rien n'est appris ni enregistré ici : son cerveau n'est pas modifié.
Tape « aide » pour la liste des commandes.
"""
import argparse

import numpy as np
import torch

from .config import Config
from .environment.keydoor_world import AGENT, DOOR, GOAL, ICE, KEY, LAVA, WALL, KeyDoorWorld
from .tasks import Task
from .train_keydoor import STEP4, load_task_agent, train
from .train_effects import EFFECT_PATH
from .train_map import MAP_PATH, MAP_V3_PATH

ARROWS = "↑↓←→"
ACTION_WORDS = {"z": 0, "haut": 0, "h": 0, "s": 1, "bas": 1, "b": 1,
                "q": 2, "gauche": 2, "g": 2, "d": 3, "droite": 3}
EVENT_WORDS = {"objectif": "goal", "o": "goal", "*": "goal",
               "clé": "key", "cle": "key", "c": "key", "k": "key",
               "porte": "door", "p": "door"}
EVENT_NAMES = {"goal": "objectif", "key": "clé", "door": "porte", "lava": "lave"}
EDITS = {"mur": WALL, "lave": LAVA, "objectif": GOAL, "clé": KEY, "cle": KEY, "porte": DOOR}

HELP = """
Commandes (Entrée seule = un pas) :
  but <événements>      donner un but, ex. « but clé porte objectif » (o/c/p acceptés)
  pas [N]               la laisser faire 1 pas (ou N pas)
  go                    la laisser agir jusqu'à réussite, mort ou 40 pas
  pense                 sa carte mentale : ce qu'elle trouve prometteur, case par case
  imagine <actions>     lui demander ce qui arriverait, ex. « imagine d d s » (z/s/q/d)
  mur|lave|objectif|clé|porte <ligne> <colonne>   poser un objet
  vide <ligne> <colonne>                          vider une case
  deplace <ligne> <colonne>                       la téléporter sur une case
  carte [N]             nouvelle carte (au hasard, ou la carte n° N)
  situation <nom>       une situation de prérequis : accessible, incontournable, ouverte, detour
                        (seul l'objectif est demandé : à elle de voir si la clé est utile)
  aide | quitter
"""


class Sandbox:
    def __init__(self, agent, cfg, seed):
        self.agent, self.cfg = agent, cfg
        self.wm, self.cost = agent.arch.world_model, agent.arch.cost
        self.map = getattr(agent.arch, "mental_map", None)
        self.world = KeyDoorWorld(cfg)
        self.rng = np.random.default_rng()
        self.new_map(seed)

    # ------------------------------------------------------------ état
    def new_map(self, seed=None):
        seed = int(self.rng.integers(10**6)) if seed is None else seed
        self.world.reset(seed=seed)
        self.seed, self.steps, self.path = seed, 0, []
        self.set_goal(("goal",))

    def set_goal(self, sequence):
        self.task = Task(" puis ".join(EVENT_NAMES[e] for e in sequence), tuple(sequence), "")
        self.agent.reset(self.task)

    def show(self):
        grid = self.world.render().split("\n")
        print("\n    " + "".join(str(c) for c in range(self.world.n)))
        for r, row in enumerate(grid):
            print(f"  {r} {row}")
        tr = self.agent.tracker
        done = [EVENT_NAMES[e] for e in self.task.sequence[:tr.progress]]
        todo = [EVENT_NAMES[e] for e in self.task.sequence[tr.progress:]]
        print(f"\n  carte n° {self.seed}   pas {self.steps}   but : {self.task.name}"
              + (f"   (fait : {', '.join(done)})" if done else "")
              + (f"   → prochain : {todo[0]}" if todo else "   ✔ but atteint"))

    # ------------------------------------------------------------ agir
    def step(self, verbose=True):
        if self.world.dead:
            print("  Elle est tombée dans la lave : « carte » pour recommencer.")
            return False
        if self.agent.tracker.done:
            print("  But atteint : donne-lui un nouveau but (« but ... »).")
            return False
        obs = self.world.observe()
        action = self.agent.act(obs)
        d = self.agent.last
        if verbose:
            self.explain(obs, d)
        _, events, dead = self.world.step(action)
        self.steps += 1
        self.path.append(ARROWS[action])
        self.agent.observe(self.world.observe())
        if verbose:
            said = ", ".join(EVENT_NAMES[e] for e in events)
            print(f"  → elle joue {ARROWS[action]}" + (f"   événement : {said}" if said else ""))
        return not dead and not self.agent.tracker.done

    @torch.no_grad()
    def explain(self, obs, d):
        """Ce qu'elle a en tête au moment de choisir."""
        target = self.agent.arch.cost.target
        plan = "".join(ARROWS[a] for a in d.chosen)
        succ = " ".join(f"{p:.2f}" for p in d.plan.success)
        dang = " ".join(f"{p:.2f}" for p in d.plan.danger)
        print(f"\n  elle vise : {EVENT_NAMES[target]}")
        plan_ev = getattr(self.agent, "plan_events", None)
        if plan_ev:
            print(f"  son plan d'événements : {' → '.join(EVENT_NAMES[e] for e in plan_ev)}"
                  f"   (~{self.agent.subgoal_cost:.0f} pas)")
        if self.map is not None:
            q = self.map.q(torch.as_tensor(obs)[None], target)[0]
            best = float(q.max()) or 1.0
            print("  carte mentale, valeur de chaque direction : "
                  + "  ".join(f"{ARROWS[a]} {float(q[a]) / best:.2f}" for a in range(4)))
        print(f"  imagination, plan retenu {plan}")
        print(f"     « {EVENT_NAMES[target]} » prévu à chaque pas : {succ}")
        print(f"     danger prévu à chaque pas       : {dang}")
        if d.plan.familiarity and d.familiarity > 0.5:
            print(f"  « déjà vu ici » : {d.familiarity:.2f} (elle cherche à aller ailleurs)")

    # ------------------------------------------------------------ lire dans sa tête
    @torch.no_grad()
    def think(self):
        if self.map is None:
            print("  Pas de carte mentale (python -m mini_iag.train_map).")
            return
        target = self.agent.tracker.next_event if not self.agent.tracker.done else self.task.sequence[-1]
        obs = torch.as_tensor(self.world.observe())[None]
        U = self.map.values(obs)[0, ("goal", "key", "door").index(target)].numpy()
        g = self.world.grid
        print(f"\n  Carte mentale pour « {EVENT_NAMES[target]} » (9 = tout près, 0 = inatteignable)")
        print("    " + "".join(str(c) for c in range(self.world.n)))
        top = U.max() or 1.0
        for r in range(self.world.n):
            row = ""
            for c in range(self.world.n):
                if (r, c) == self.world.agent:
                    row += "A"
                elif g[WALL, r, c]:
                    row += "#"
                elif g[LAVA, r, c]:
                    row += "~"
                else:
                    row += str(min(9, int(10 * U[r, c] / top)))
            print(f"  {r} {row}")

    @torch.no_grad()
    def imagine(self, words):
        actions = [ACTION_WORDS[w] for w in words if w in ACTION_WORDS]
        if not actions:
            print("  Exemple : imagine d d s   (z haut, s bas, q gauche, d droite)")
            return
        z = self.wm.encode(torch.as_tensor(self.world.observe())[None])
        traj = self.wm.rollout(z, torch.tensor([actions]))[0]
        prev = torch.cat([z, traj[:-1]])
        p = self.cost.events(torch.cat([prev, traj], -1))
        print(f"\n  Elle imagine {''.join(ARROWS[a] for a in actions)} sans bouger :")
        for t, a in enumerate(actions):
            said = [f"{EVENT_NAMES[e]} {float(p[t, i]):.0%}" for i, e in enumerate(("goal", "key", "door", "lava"))
                    if p[t, i] > 0.1]
            print(f"   pas {t + 1} {ARROWS[a]} : " + (", ".join(said) if said else "rien de spécial"))
        print("  (c'est sa prédiction ; tape les mêmes coups avec « pas » pour comparer à la réalité)")

    # ------------------------------------------------------------ modifier le monde
    def edit(self, what, r, c):
        n, w = self.world.n, self.world
        if not (0 < r < n - 1 and 0 < c < n - 1):
            print(f"  Case hors de la carte (lignes et colonnes de 1 à {n - 2}).")
            return
        if what == "deplace":
            if w.grid[WALL, r, c] or (w.grid[DOOR, r, c] and not w.has_key):
                print("  Case occupée.")
                return
            w.agent = (r, c)
        else:
            if what != "vide" and (r, c) == w.agent:
                print("  Elle est sur cette case.")
                return
            w.grid[[WALL, LAVA, GOAL, KEY, DOOR], r, c] = 0
            if w.n_ice:
                w.ice[r, c] = 0
            if what != "vide":
                w.grid[EDITS[what], r, c] = 1
        self.agent.prev, self.agent._fresh = None, False   # une modification n'est pas un événement vécu
        self.agent.core.reset()


def main():
    p = argparse.ArgumentParser(prog="python -m mini_iag.bac_a_sable")
    p.add_argument("--carte", type=int, default=None)
    p.add_argument("--sans-carte", action="store_true", help="sans la carte mentale")
    p.add_argument("--sans-prerequis", action="store_true",
                   help="version du verdict (sous-buts fournis), même si la version des lots 2-3 existe")
    args = p.parse_args()
    if not STEP4.exists():
        print("Pas encore de cerveau : éducation de l'étape 4a (~6 minutes)...")
        train(verbose=False)
    use_map = MAP_PATH.exists() and not args.sans_carte
    if not MAP_PATH.exists():
        print("(Pas de carte mentale : python -m mini_iag.train_map. Elle joue sans.)")
    if MAP_V3_PATH.exists() and EFFECT_PATH.exists() and not args.sans_carte and not args.sans_prerequis:
        from .final_agent import build_prerequisite_agent
        agent = build_prerequisite_agent(STEP4, MAP_V3_PATH, EFFECT_PATH)
        print("(Version des lots 2-3 : elle cherche elle-même ses sous-buts.)")
    else:
        agent = load_task_agent(mental_map=MAP_PATH if use_map else None)
    sb = Sandbox(agent, Config.keydoor(), args.carte)
    print(__doc__.split("\n\n")[1])
    print(HELP)
    sb.show()
    while True:
        try:
            line = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        words = line.split()
        cmd, rest = (words[0], words[1:]) if words else ("pas", [])
        try:
            if cmd in ("quitter", "q!", "exit"):
                break
            elif cmd == "aide":
                print(HELP)
                continue
            elif cmd == "pas":
                for _ in range(int(rest[0]) if rest else 1):
                    if not sb.step(verbose=not rest or int(rest[0]) == 1):
                        break
            elif cmd == "go":
                start = len(sb.path)
                for _ in range(40):
                    if not sb.step(verbose=False):
                        break
                print(f"  chemin : {''.join(sb.path[start:])}   "
                      + ("✔ but atteint" if sb.agent.tracker.done else
                         "✘ lave" if sb.world.dead else "… pas encore"))
            elif cmd == "but":
                seq = [EVENT_WORDS[w] for w in rest if w in EVENT_WORDS]
                if not seq:
                    print("  Exemple : but clé porte   (objectif, clé, porte)")
                    continue
                sb.set_goal(seq)
            elif cmd == "pense":
                sb.think()
                continue
            elif cmd == "imagine":
                sb.imagine(rest)
                continue
            elif cmd in EDITS or cmd in ("vide", "deplace"):
                sb.edit(cmd, int(rest[0]), int(rest[1]))
            elif cmd == "carte":
                sb.new_map(int(rest[0]) if rest else None)
            elif cmd == "situation":
                from .environment.situations import SITUATIONS, make_situation
                name = rest[0] if rest and rest[0] in SITUATIONS else "incontournable"
                sb.seed = make_situation(sb.world, name, int(sb.rng.integers(10**6)))
                sb.steps, sb.path = 0, []
                sb.set_goal(("goal",))
            else:
                print("  Commande inconnue (« aide »).")
                continue
        except (ValueError, IndexError):
            print("  Commande mal formée (« aide »).")
            continue
        sb.show()


if __name__ == "__main__":
    main()
