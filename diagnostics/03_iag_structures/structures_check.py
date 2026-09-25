"""
structures_check.py — Mini-IAG, étape 1 : l'architecture vide.

Question : à l'initialisation aléatoire, que sait faire chaque module ?
La feuille de route dit "le système ne produit que du bruit". On le mesure,
module par module, pour avoir des chiffres de référence (le "avant") à
comparer après l'entraînement de l'étape 2.

Données : transitions (obs, action, obs suivante) collectées par un agent
qui marche au hasard sur 300 cartes aléatoires.

Mesures :
  S1  modèle du monde : connaissant z_t et z_{t+1}, peut-il dire QUELLE action
      a été faite ? On prédit l'état suivant pour chacune des 4 actions et on
      garde la plus proche. (Deux actions bloquées par un mur donnent le même
      résultat : elles comptent toutes deux comme justes.)
  S2  information dans le latent : peut-on retrouver la case de l'agent à partir
      du vecteur latent par une simple régression linéaire (sonde) ?
  S3  module de coût : AUC pour détecter la lave et l'objectif (0.5 = hasard)
  S4  espace de travail : parmi 8 contenus dont un seul est pertinent,
      l'attention se porte-t-elle sur le bon ? (hasard = 12.5 %)
  S5  mémoire : retrouve-t-elle un état déjà stocké ?
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mini_iag import Architecture, Config  # noqa: E402
from mini_iag.data import collect  # noqa: E402
from mini_iag.metrics import (action_identification, agent_cell, auc,  # noqa: E402
                              labels, linear_probe, workspace_selection)


if __name__ == "__main__":
    cfg = Config()
    arch = Architecture(cfg)
    arch.eval()
    data = collect(cfg)
    obs, act, nxt = data.obs, data.action, data.next_obs
    lava, goal = labels(nxt)

    with torch.no_grad():
        wm = arch.world_model
        act_acc, act_chance, err_pred, err_copy = action_identification(wm, obs, act, nxt)
        z_all = wm.encode(nxt)
        cell, n_cells = agent_cell(nxt)
        probe_latent = linear_probe(z_all, cell, n_cells)
        probe_raw = linear_probe(nxt.reshape(len(nxt), -1), cell, n_cells)
        chance = torch.bincount(cell, minlength=n_cells).max().item() / len(cell)
        danger, success, _ = arch.cost(z_all)
        auc_lava, auc_goal = auc(danger, lava), auc(success, goal)
        ws_acc, ws_att, ws_ent = workspace_selection(arch.workspace, cfg.latent_dim)
        arch.memory.write(z_all[:1000])
        _, idx = arch.memory.recall(z_all[:1000], k=1)
        # un même état peut apparaître plusieurs fois : on compare les contenus
        mem_ok = (z_all[idx[:, 0]] - z_all[:1000]).abs().max(1).values.lt(1e-5).float().mean().item()

    print(f"{len(obs)} transitions, {300} cartes aléatoires, poids aléatoires (graine {cfg.seed})\n")
    print("Paramètres :")
    for k, v in arch.parameter_counts().items():
        print(f"  {k:34} {v:>8,}")
    print("\nS1 modèle du monde")
    print(f"   action retrouvée            {100 * act_acc:.1f} %   (hasard {100 * act_chance:.1f} %)")
    print(f"   erreur de prédiction        {err_pred:.2e}")
    print(f"   erreur 'rien ne change'     {err_copy:.2e}   (le latent aléatoire bouge à peine)")
    print("S2 case de l'agent retrouvée par une sonde linéaire")
    print(f"   depuis le latent aléatoire  {100 * probe_latent:.1f} %")
    print(f"   depuis l'image brute        {100 * probe_raw:.1f} %")
    print(f"   hasard (classe majoritaire) {100 * chance:.1f} %")
    print("S3 module de coût (AUC, 0.5 = hasard)")
    print(f"   détection de la lave        {auc_lava:.2f}   ({int(lava.sum())} états sur lave)")
    print(f"   détection de l'objectif     {auc_goal:.2f}   ({int(goal.sum())} états sur l'objectif)")
    print("S4 espace de travail (1 contenu pertinent parmi 8)")
    print(f"   attention sur le bon        {ws_att:.3f}   (uniforme 0.125)")
    print(f"   entropie normalisée         {ws_ent:.3f}   (1 = attention uniforme)")
    print(f"   argmax sur le bon           {100 * ws_acc:.1f} %   (trompeur : écarts minuscules, voir README)")
    print("S5 mémoire")
    print(f"   état stocké retrouvé        {100 * mem_ok:.1f} %")
