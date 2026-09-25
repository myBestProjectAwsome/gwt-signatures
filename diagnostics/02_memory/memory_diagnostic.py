"""
memory_diagnostic.py — Diagnostic de la mémoire épisodique.

Question : la mémoire UTILISE-t-elle vraiment le contenu diffusé par le
workspace, ou produit-elle des souvenirs au hasard ?

Condition normale : psyche complet (monde simulé, messages variés).
Ablation 1 "broadcast brouillé" : la mémoire reçoit, à la place du vrai
contenu conscient, un contenu passé tiré au hasard. Tout le reste est
identique. Si la mémoire dépend du broadcast, D1 et D2 doivent s'effondrer.
Ablation 2 "saillance ignorée" : la mémoire reçoit tous les contenus avec
la même saillance (0.5). Cela aplatit à la fois l'importance à l'encodage et
la force de l'indice. Si la pondération par saillance compte, D3 (et la
réaction aux messages) doivent s'effondrer.

Mesures :
  D1  pertinence : similarité entre l'indice (contenu du cycle précédent)
      et le souvenir rappelé, comparée à un souvenir quelconque (hasard)
  D2  coordination : P(la mémoire gagne au cycle suivant | l'explorateur a
      diffusé "revoir un souvenir"), et | un message utilisateur
  D3  importance : les épisodes utilisateur sont-ils sur-représentés dans
      les rappels par rapport à leur fréquence dans l'expérience vécue ?
  D4  persistance : dans une 2e session (même fichier SQLite, nouveaux
      objets), des souvenirs de la 1re session sont-ils rappelés ?
"""
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from psyche import FakeEnvironment, MemoryStore, Proposal, build, step  # noqa: E402
from psyche.modules import EpisodicMemory  # noqa: E402

CYCLES = 1000
SEEDS = [0, 1, 2]
MESSAGES_A = ("bonjour", "salut psyche", "tu fais quoi ?", "je travaille sur mon projet",
              "bonne nuit", "bonjour psyche", "à demain")
MESSAGES_B = ("rebonjour", "je suis revenu", "tu te souviens de moi ?",
              "je travaille encore sur mon projet", "bonne soirée")
PREFIXES = ("cela me rappelle : ", "je me souviens : ")


class ScrambledMemory(EpisodicMemory):
    """Ablation : reçoit un contenu passé au hasard au lieu du vrai broadcast."""

    def __init__(self, store, seed):
        super().__init__(store)
        self.rng = random.Random(seed + 1000)
        self.seen = []

    def receive(self, broadcast, h):
        if broadcast.source == self.name:
            return super().receive(broadcast, h)
        self.seen.append(broadcast)
        fake = self.rng.choice(self.seen)
        super().receive(Proposal(fake.source, fake.content, fake.salience), h)


class IgnoreSalienceMemory(EpisodicMemory):
    """Ablation : ignore la saillance (importance ET force de l'indice = 0.5)."""

    def receive(self, broadcast, h):
        if broadcast.source == self.name:
            return super().receive(broadcast, h)
        super().receive(Proposal(broadcast.source, broadcast.content, 0.5), h)


def recalled_text(content):
    for p in PREFIXES:
        if content.startswith(p):
            return content[len(p):]
    return content


def simulate(seed, ablation=None, cycles=CYCLES, store=None, messages=MESSAGES_A):
    random.seed(seed)
    env = FakeEnvironment(seed=seed, user_period=25, user_messages=messages)
    store = store if store is not None else MemoryStore()
    h, ws, user, modules = build(env, store=store, memory=False)
    if ablation == "brouillé":
        mem = ScrambledMemory(store, seed)
    elif ablation == "saillance ignorée":
        mem = IgnoreSalienceMemory(store)
    else:
        mem = EpisodicMemory(store)
    modules.append(mem)
    enc = store.encoder
    clock_start = store.clock
    winners, rows = [], []
    for t in range(cycles):
        # baseline D1 : similarité moyenne indice / souvenir quelconque
        prev = winners[-1] if winners else None
        winner, _ = step(env, h, ws, user, modules)
        env.tick()
        winners.append(winner)
        row = dict(t=t, source=winner.source, content=winner.content)
        if winner.source == "memoire" and prev is not None and prev.source != "memoire":
            txt = recalled_text(winner.content)
            ep = next(e for e in store.episodes() if e.content == txt)
            cue_vec = enc.encode(prev.content)
            others = [e for e in store.episodes() if e.content != prev.content]
            row.update(
                cue_kind=winner.content.split(" : ")[0],
                sim=float(enc.encode(txt) @ cue_vec),
                base=float(np.mean([enc.encode(e.content) @ cue_vec for e in others])),
                rec_source=ep.source,
                from_past_session=ep.first_episode <= clock_start,
            )
        elif winner.source == "memoire":
            txt = recalled_text(winner.content)
            ep = next(e for e in store.episodes() if e.content == txt)
            row.update(rec_source=ep.source, from_past_session=ep.first_episode <= clock_start)
        rows.append(row)
    return rows, store


def conditional(rows, pred):
    """P(mémoire gagne en t+1 | pred(gagnant en t))."""
    hits = [rows[i + 1]["source"] == "memoire" for i in range(len(rows) - 1) if pred(rows[i])]
    return 100 * np.mean(hits) if hits else float("nan"), len(hits)


def analyse(all_rows, stores):
    recalls = [r for r in all_rows if "sim" in r and r["cue_kind"] == "cela me rappelle"]
    mem_wins = [r for r in all_rows if r["source"] == "memoire"]
    is_revoir = lambda r: r["source"] == "exploration" and r["content"] == "revoir un souvenir"
    is_user = lambda r: r["source"] == "utilisateur"
    lived = sum(e.count for s in stores for e in s.episodes())
    lived_user = sum(e.count for s in stores for e in s.episodes() if e.source == "utilisateur")
    return dict(
        sim=np.mean([r["sim"] for r in recalls]) if recalls else float("nan"),
        base=np.mean([r["base"] for r in recalls]) if recalls else float("nan"),
        n_recalls=len(recalls),
        after_revoir=conditional(all_rows, is_revoir),
        after_user=conditional(all_rows, is_user),
        base_rate=100 * np.mean([r["source"] == "memoire" for r in all_rows]),
        user_in_lived=100 * lived_user / lived,
        user_in_recalls=100 * np.mean([r["rec_source"] == "utilisateur" for r in mem_wins]),
    )


def persistence(seed=0):
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "episodes.sqlite"
        s1 = MemoryStore(db)
        simulate(seed, store=s1, messages=MESSAGES_A)
        n1 = len(s1)
        s1.close()
        s2 = MemoryStore(db)                    # nouvelle session : on relit le fichier
        rows2, _ = simulate(seed + 7, store=s2, messages=MESSAGES_B)
        mem = [r for r in rows2 if r["source"] == "memoire"]
        past = [r for r in mem if r["from_past_session"]]
        examples = sorted({recalled_text(r["content"]) for r in past
                           if "utilisateur" in r["content"]})
        s2.close()
    return dict(n1=n1, n_mem=len(mem), n_past=len(past), examples=examples)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CONDS = ["normal", "brouillé", "saillance ignorée"]
    COLORS = {"normal": "tab:blue", "brouillé": "tab:gray", "saillance ignorée": "tab:orange"}
    res = {}
    for cond in CONDS:
        runs = [simulate(s, ablation=None if cond == "normal" else cond) for s in SEEDS]
        res[cond] = analyse([r for rows, _ in runs for r in rows], [st for _, st in runs])
    pers = persistence()

    print(f"{CYCLES} cycles x {len(SEEDS)} graines, monde simulé, messages variés\n")
    print(f"{'':30}" + "".join(f"{c:>21}" for c in CONDS))
    fmt = lambda v, pct: f"{v:20.0f}%" if pct else f"{v:21.2f}"
    lines = [("D1 similarité indice/rappel", lambda r: r["sim"], False),
             ("   (souvenir au hasard)", lambda r: r["base"], False),
             ("D2 P(mém | revoir souvenir)", lambda r: r["after_revoir"][0], True),
             ("   P(mém | message)", lambda r: r["after_user"][0], True),
             ("   P(mém) tous cycles", lambda r: r["base_rate"], True),
             ("D3 part utilisateur, vécu", lambda r: r["user_in_lived"], True),
             ("   part utilisateur, rappels", lambda r: r["user_in_recalls"], True)]
    for label, f, pct in lines:
        print(f"{label:30}" + "".join(fmt(f(res[c]), pct) for c in CONDS))
    print(f"\nD4 persistance : session 1 -> {pers['n1']} souvenirs sur disque")
    print(f"   session 2 : {pers['n_past']}/{pers['n_mem']} rappels viennent de la session 1")
    print(f"   ex. : {pers['examples'][:4]}")

    # ------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
    w = 0.27

    a = ax[0]
    x = np.arange(len(CONDS))
    a.bar(x - 0.2, [res[c]["sim"] for c in CONDS], 0.4, color=[COLORS[c] for c in CONDS],
          label="souvenir rappelé")
    a.bar(x + 0.2, [res[c]["base"] for c in CONDS], 0.4, color="white", edgecolor="black",
          hatch="//", label="souvenir au hasard")
    a.set_xticks(x, [c.replace(" ", "\n") for c in CONDS], fontsize=8)
    a.set(title="D1 — Pertinence du rappel", ylabel="similarité avec l'indice", ylim=(0, 1))
    a.legend(fontsize=8)

    a = ax[1]
    labels = ["après\n« revoir un souvenir »", "après\nun message", "n'importe\nquel cycle"]
    x = np.arange(3)
    for i, c in enumerate(CONDS):
        vals = [res[c]["after_revoir"][0], res[c]["after_user"][0], res[c]["base_rate"]]
        a.bar(x + (i - 1) * w, vals, w, color=COLORS[c], label=c)
    a.set_xticks(x, labels, fontsize=8)
    a.set(title="D2 — La mémoire gagne au cycle suivant", ylabel="%", ylim=(0, 105))
    a.legend(fontsize=8)

    a = ax[2]
    x = np.arange(2)
    for i, c in enumerate(CONDS):
        a.bar(x + (i - 1) * w, [res[c]["user_in_lived"], res[c]["user_in_recalls"]],
              w, color=COLORS[c], label=c)
    a.set_xticks(x, ["dans le vécu", "dans les rappels"])
    a.set(title="D3 — Part des épisodes utilisateur", ylabel="%")
    a.legend(fontsize=8)

    a = ax[3]
    a.bar(["session 1\n(sur disque)", "session 2"],
          [pers["n_past"], pers["n_mem"] - pers["n_past"]],
          color=["tab:purple", "tab:blue"])
    a.set(title="D4 — Origine des souvenirs rappelés\nen session 2", ylabel="nb de rappels")

    fig.tight_layout()
    out = Path(__file__).with_name("memory_results.png")
    fig.savefig(out, dpi=120)
    print(f"\nFigure : {out.name}")
