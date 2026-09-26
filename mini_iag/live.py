"""La mini-IAG vit sur ton ordinateur.   python -m mini_iag.live

Elle joue carte après carte, apprend en vivant, et sauvegarde tout
(checkpoints/life.pt) : relance la commande, elle reprend où elle en était.

Pendant qu'elle vit, tape une commande puis Entrée :
    o   impose la tâche « objectif »          c   impose la tâche « clé »
    oc  impose « objectif puis clé »          a   tâches au hasard (par défaut)
    r   regarder pas à pas (r à nouveau pour arrêter)
    b   bilan détaillé                        q   sauvegarder et quitter
Ctrl+C sauvegarde aussi avant de quitter.

Options :  --regarder (pas à pas dès le départ)  --delai 0.3  --episodes N
           --bilan (affiche la progression et trace checkpoints/life_progress.png)
           --oublier (repart de l'état de l'étape 4a ; l'ancienne vie est gardée en .bak)

Sécurité : elle vit dans son monde virtuel. Elle n'écrit que dans checkpoints/.
"""
import argparse
import sys
import time
from pathlib import Path

from .life import LIFE_PATH, LIFE_TASKS, Life
from .train_keydoor import STEP4, load_task_agent, train

ARROWS = "↑↓←→"
COMMANDS = {"o": LIFE_TASKS[0], "c": LIFE_TASKS[1], "oc": LIFE_TASKS[2], "a": None}


def read_command():
    """Commande tapée au clavier, sans bloquer (Linux/macOS)."""
    try:
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip().lower()
    except (ImportError, OSError, ValueError):
        pass
    return None


def bar(pct, width=20):
    n = int(round(pct / 100 * width))
    return "█" * n + "·" * (width - n)


def dashboard(life):
    r = life.recent(100)
    print("\n" + "─" * 64)
    print(f" Épisodes vécus : {life.episode}    séances d'apprentissage : {life.learner.n_updates}"
          f"    souvenirs récents : {len(life.buffer)} pas")
    if r:
        print(f" 100 derniers   succès {bar(r['succès'])} {r['succès']:5.1f} %"
              f"   lave {r['lave']:4.1f} %   bloqué {r['bloqué']:4.1f} %")
        for t in LIFE_TASKS:
            rt = life.recent(100, t.name)
            if rt:
                print(f"   {t.name:18}  succès {rt['succès']:5.1f} %  (sur {rt['n']})")
        first = [x["surprise"] for x in life.history[:100]]
        last = [x["surprise"] for x in life.history[-100:]]
        print(f" Surprise moyenne : {sum(first) / len(first):.1f} au début -> "
              f"{sum(last) / len(last):.1f} maintenant")
    task = life.forced_task.name if life.forced_task else "au hasard"
    print(f" Tâche : {task}.   Commandes : o, c, oc, a, r (regarder), b (bilan), q (quitter)")
    print("─" * 64)


def progress_plot(life, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    h = life.history
    if len(h) < 50:
        print("Pas assez d'épisodes pour une courbe (il en faut au moins 50).")
        return
    w = 100 if len(h) >= 200 else 25
    ep = np.arange(1, len(h) + 1)
    roll = lambda k: np.convolve([x["outcome"] == k for x in h], np.ones(w) / w, "valid") * 100
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for k, c in (("succès", "#2a78d6"), ("lave", "#eb6834"), ("bloqué", "#8a8980")):
        ax[0].plot(ep[w - 1:], roll(k), color=c, lw=2, label=k)
    ax[0].set(title=f"Issues (moyenne glissante sur {w} épisodes)", xlabel="épisode vécu",
              ylabel="%", ylim=(0, 100))
    ax[0].legend(frameon=False)
    s = np.array([x["surprise"] for x in h])
    ax[1].plot(ep[w - 1:], np.convolve(s, np.ones(w) / w, "valid"), color="#1baf7a", lw=2)
    ax[1].set(title="Surprise (écart entre imagination et réalité)", xlabel="épisode vécu")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e6e5df")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Courbe de progression : {path}")


def main():
    p = argparse.ArgumentParser(prog="python -m mini_iag.live")
    p.add_argument("--regarder", action="store_true")
    p.add_argument("--delai", type=float, default=0.3)
    p.add_argument("--episodes", type=int, default=0, help="0 = sans fin")
    p.add_argument("--bilan", action="store_true")
    p.add_argument("--oublier", action="store_true")
    args = p.parse_args()

    if not STEP4.exists():
        print("Pas encore de cerveau : éducation de l'étape 4a (~6 minutes)...")
        train(verbose=False)
    if args.oublier and LIFE_PATH.exists():
        LIFE_PATH.rename(LIFE_PATH.with_suffix(".pt.bak"))
        print("Nouvelle vie : l'ancienne est gardée dans life.pt.bak")
    print("Réveil de la mini-IAG...")
    life = Life(load_task_agent())
    print("Elle reprend sa vie." if life.load() else "Première vie : elle part de l'étape 4a.")

    if args.bilan:
        dashboard(life)
        progress_plot(life, Path("checkpoints/life_progress.png"))
        return

    watch = {"on": args.regarder}

    def on_step(world, agent, t):
        if not watch["on"]:
            return
        d = agent.last
        plan = "".join(ARROWS[a] for a in d.chosen)
        print("\033[2J\033[H" if sys.stdout.isatty() else "", end="")
        print(f"épisode {life.episode + 1} — tâche « {agent.tracker.task.name} » — pas {t}\n")
        print(world.render())
        print(f"\n  plan imaginé {plan}   succès prévu {max(d.plan.success):.2f}"
              f"   danger prévu {max(d.plan.danger):.2f}")
        print("  (# mur  ~ lave  * objectif  k clé  D porte  A agent, a = avec la clé)")
        time.sleep(args.delai)

    dashboard(life)
    try:
        while args.episodes == 0 or life.episode < args.episodes:
            cmd = read_command()
            if cmd in COMMANDS:
                life.forced_task = COMMANDS[cmd]
                print(f"→ tâche : {COMMANDS[cmd].name if COMMANDS[cmd] else 'au hasard'}")
            elif cmd == "r":
                watch["on"] = not watch["on"]
            elif cmd == "b":
                dashboard(life)
            elif cmd == "q":
                break
            r = life.live_one(on_step)
            mark = {"succès": "✔", "lave": "✘ lave", "bloqué": "… bloqué"}[r["outcome"]]
            learned = "  (+ séance d'apprentissage)" if r["loss"] is not None else ""
            print(f"épisode {r['episode']:5d}  {r['task']:18} {mark:9} en {r['steps']:2d} pas"
                  f"   surprise {r['surprise']:5.1f}{learned}")
            if life.episode % 25 == 0:
                life.save()
            if life.episode % 50 == 0:
                dashboard(life)
    except KeyboardInterrupt:
        print("\nInterruption demandée.")
    life.save()
    print(f"Sauvegardé : {life.episode} épisodes vécus. Relance la commande pour qu'elle reprenne.")


if __name__ == "__main__":
    main()
