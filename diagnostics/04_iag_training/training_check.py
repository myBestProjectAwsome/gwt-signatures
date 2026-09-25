"""
training_check.py — Mini-IAG, étape 2 : l'éducation des modules.

Question : après entraînement, chaque module a-t-il acquis sa compétence,
sur des cartes JAMAIS VUES ? Et chaque ingrédient ajouté au modèle du monde
est-il nécessaire (ablations) ?

Conditions :
  avant           poids aléatoires (état de l'étape 1)
  complet         JEPA + dynamique inverse + événements + entraînement multi-pas
  sans inverse    idem sans la tête de dynamique inverse     (réentraîné)
  sans événements idem sans la tête d'événements             (réentraîné)
  sans multi-pas  idem entraîné sur 1 pas seulement          (réentraîné)
Pour chaque variante, le module de coût est réentraîné sur SES latents.

Mesures (cartes de test, graine 1) :
  S1  action retrouvée entre deux états (hasard ~30 %)
  S2  position de l'agent / de l'objectif lisible dans le latent (sonde linéaire)
  S3  anticipation : AUC du coût sur les états IMAGINÉS 1 à 5 pas à l'avance
  S4  workspace : attention sur le contenu signalé parmi 8 (hasard 0.125) et
      case retrouvée par une lecture qui ne voit que les slots

Options : --quick (sans les ablations ni figure, ~20 s), sinon quelques minutes.
          --replot (redessine la figure depuis training_results.json).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mini_iag import Architecture, Config  # noqa: E402
from mini_iag.data import (TEST_SEED, TRAIN_SEED, collect, collect_event_sequences,  # noqa: E402
                           collect_segments)
from mini_iag.environment.gridworld import AGENT, GOAL  # noqa: E402
from mini_iag.metrics import action_identification, auc, linear_probe  # noqa: E402
from mini_iag.train import TRAIN_MAPS, load, train  # noqa: E402
from mini_iag.training import (CostTrainer, SelectionReadout, WorkspaceTrainer,  # noqa: E402
                               WorldModelTrainer)

CKPT = ROOT / "checkpoints" / "step2.pt"
HORIZONS = 5
VARIANTS = {"sans inverse": dict(inv_weight=0.0),
            "sans événements": dict(event_weight=0.0),
            "sans multi-pas": dict(multistep_weight=0.0)}


@torch.no_grad()
def evaluate_world(arch, test, seqs, probe_obs):
    wm = arch.world_model
    s1 = action_identification(wm, test.obs, test.action, test.next_obs)[0]
    z = wm.target_encoder(probe_obs)
    cell = lambda ch: probe_obs[:, ch].reshape(len(probe_obs), -1).argmax(1)
    s0, acts, ev, alive = seqs
    roll = wm.rollout(wm.encode(s0), acts)
    lava, goal = [], []
    for h in range(HORIZONS):
        m = alive[:, h]
        d, s, _ = arch.cost(roll[m, h])
        lava.append(auc(d, ev[m, h, 0]))
        goal.append(auc(s, ev[m, h, 1]))
    return dict(S1=s1, agent=linear_probe(z, cell(AGENT), 49),
                goal_pos=linear_probe(z, cell(GOAL), 49), lava=lava, goal=goal)


def train_variant(cfg, train_data, segments, **weights):
    torch.manual_seed(0)
    arch = Architecture(cfg)
    WorldModelTrainer(arch.world_model, **weights).fit(train_data, None, segments, verbose=False)
    for p in arch.world_model.parameters():
        p.requires_grad = False
    CostTrainer(arch.cost, arch.world_model).fit(train_data, None, verbose=False)
    return arch


def workspace_conditions(cfg, arch, readout, train_data, test_data):
    """avant (tout aléatoire), workspace gelé + lecture entraînée, complet."""
    out = {}
    fresh = Architecture(cfg)
    wt = WorkspaceTrainer(fresh.workspace, SelectionReadout(cfg), arch.world_model)
    out["avant"] = wt.evaluate(wt.prepare(test_data.next_obs))
    torch.manual_seed(0)
    frozen = Architecture(cfg)
    wt = WorkspaceTrainer(frozen.workspace, SelectionReadout(cfg), arch.world_model,
                          train_workspace=False)
    wt.fit(train_data.next_obs, verbose=False)
    out["workspace gelé"] = wt.evaluate(wt.prepare(test_data.next_obs))
    wt = WorkspaceTrainer(arch.workspace, readout, arch.world_model)
    out["complet"] = wt.evaluate(wt.prepare(test_data.next_obs))
    return out


def figure(results, ws, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
    COLORS = {"complet": "#2a78d6", "sans inverse": "#eb6834",
              "sans événements": "#1baf7a", "sans multi-pas": "#eda100"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    hs = list(range(1, HORIZONS + 1))
    for a, key, title in [(ax[0], "lava", "S3 — Anticiper la lave"),
                          (ax[1], "goal", "S3 — Anticiper l'objectif")]:
        a.plot(hs, results["avant"][key], "--", color="#8a8980", lw=2)
        a.text(HORIZONS + 0.1, results["avant"][key][-1], "avant", color=MUTED, va="center")
        ends = []
        for name, col in COLORS.items():
            if name not in results:
                continue
            a.plot(hs, results[name][key], "-o", color=col, lw=2, ms=8,
                   markeredgecolor="white", markeredgewidth=2, label=name)
            ends.append((results[name][key][-1], name, col))
        ends.sort()
        last = -1
        for y, name, col in ends:                      # étiquettes directes sans chevauchement
            y = max(y, last + 0.035)
            a.text(HORIZONS + 0.1, y, name, color=INK, va="center")
            a.plot([HORIZONS + 0.02, HORIZONS + 0.08], [y, y], color=col, lw=2)
            last = y
        a.set(title=title, xlabel="pas imaginés à l'avance", ylabel="AUC (0,5 = hasard)",
              ylim=(0.4, 1.02), xlim=(0.8, HORIZONS + 1.1), xticks=hs)
        a.grid(axis="y", color=GRID)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(loc="lower left", fontsize=8, frameon=False)
    a = ax[2]
    names = ["avant", "workspace gelé", "complet"]
    vals = [ws[n]["attention"] for n in names]
    bars = a.bar(names, vals, color=["#8a8980", "#86b6ef", "#2a78d6"], width=0.6,
                 edgecolor="white", linewidth=2)
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", color=INK)
    a.axhline(0.125, color=MUTED, lw=1, ls=":")
    a.set(title="S4 — Attention sur le contenu signalé\n(hasard : 0,125, ligne pointillée)",
          ylim=(0, 1.1), ylabel="masse d'attention")
    a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="sans les ablations")
    parser.add_argument("--replot", action="store_true", help="redessine depuis le JSON")
    args = parser.parse_args()
    out = Path(__file__).with_name("training_results.png")
    saved = out.with_suffix(".json")
    if args.replot:
        data = json.loads(saved.read_text())
        figure(data["world"], data["workspace"], out)
        print(f"Figure : {out.name}")
        sys.exit()

    cfg = Config()
    if not CKPT.exists():
        print("Pas de checkpoint : entraînement (python -m mini_iag.train)...")
        train(CKPT, verbose=False)
    arch, readout, _ = load(CKPT)

    train_data = collect(cfg, n_maps=TRAIN_MAPS, seed=TRAIN_SEED)
    test_data = collect(cfg, n_maps=300, seed=TEST_SEED)
    seqs = collect_event_sequences(cfg, n_seq=6000)
    probe_obs = collect(cfg, n_maps=1500, seed=TEST_SEED).next_obs

    torch.manual_seed(0)
    before = Architecture(cfg)
    results = {"avant": evaluate_world(before, test_data, seqs, probe_obs),
               "complet": evaluate_world(arch, test_data, seqs, probe_obs)}
    if not args.quick:
        segments = collect_segments(cfg, n_maps=TRAIN_MAPS, seed=TRAIN_SEED)
        for name, w in VARIANTS.items():
            print(f"ablation : {name} (réentraînement)...", flush=True)
            results[name] = evaluate_world(train_variant(cfg, train_data, segments, **w),
                                           test_data, seqs, probe_obs)
    ws = workspace_conditions(cfg, arch, readout, train_data, test_data)

    print(f"\nCartes de test jamais vues (graine {TEST_SEED})\n")
    names = list(results)
    print(f"{'':30}" + "".join(f"{n:>17}" for n in names))
    rows = [("S1 action retrouvée", lambda r: f"{100 * r['S1']:.1f} %"),
            ("S2 position agent (sonde)", lambda r: f"{100 * r['agent']:.1f} %"),
            ("   position objectif (sonde)", lambda r: f"{100 * r['goal_pos']:.1f} %")]
    for h in (0, 2, 4):
        rows.append((f"S3 lave, {h + 1} pas à l'avance", lambda r, h=h: f"{r['lava'][h]:.3f}"))
    for h in (0, 2, 4):
        rows.append((f"   objectif, {h + 1} pas", lambda r, h=h: f"{r['goal'][h]:.3f}"))
    for label, f in rows:
        print(f"{label:30}" + "".join(f"{f(results[n]):>17}" for n in names))
    print("\nS4 espace de travail (1 contenu signalé parmi 8 états réels)")
    for n, r in ws.items():
        print(f"   {n:16} attention {r['attention']:.3f}   case retrouvée {100 * r['cell_acc']:.1f} %")

    if args.quick:
        # la figure du README contient les ablations : --quick ne doit pas l'écraser
        print("\n(--quick : figure non régénérée ; lancer sans --quick pour la mettre à jour)")
    else:
        saved.write_text(json.dumps({"world": results, "workspace": ws}, indent=1))
        figure(results, ws, out)
        print(f"\nFigure : {out.name}")
