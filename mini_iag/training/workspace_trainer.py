"""Entraînement de l'espace de travail (étape 2c) : l'attention sélective.

Tâche ("trier les flux pour ne garder que l'essentiel") :
  - 8 contenus arrivent en même temps : les latents de 8 états RÉELS tirés
    de cartes différentes (donc tous plausibles, aucun n'est du bruit) ;
  - un seul est signalé par un indice (un vecteur fixe ajouté, comme si un
    autre module le marquait comme pertinent) ;
  - une tête de lecture qui ne voit QUE les 2 slots doit dire ce que contient
    le contenu signalé (case de l'agent, danger, succès).

L'attention n'est JAMAIS supervisée : seule la réussite de la lecture l'est.
Si le workspace apprend à laisser entrer le contenu signalé, c'est parce que
le goulot l'y oblige.
"""
import torch
from torch.nn import functional as F

from ..metrics import agent_cell, labels


def make_selection_batch(latents, cells, flags, cue, n_tokens, batch, gen):
    idx = torch.randint(len(latents), (batch, n_tokens), generator=gen)
    target = torch.randint(n_tokens, (batch,), generator=gen)
    tokens = latents[idx].clone()
    rows = torch.arange(batch)
    tokens[rows, target] += cue
    t_idx = idx[rows, target]
    return tokens, target, cells[t_idx], flags[t_idx]


class WorkspaceTrainer:
    N_TOKENS = 8

    def __init__(self, workspace, readout, world_model, lr=2e-3, iters=3000,
                 batch_size=256, cue_norm=3.0, seed=0, train_workspace=True):
        self.ws, self.readout, self.wm = workspace, readout, world_model
        self.iters, self.batch_size = iters, batch_size
        g = torch.Generator().manual_seed(seed + 99)
        cue = torch.randn(workspace.slots.shape[1], generator=g)
        self.cue = cue / cue.norm() * cue_norm
        params = list(readout.parameters())
        if train_workspace:
            params += list(workspace.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.gen = torch.Generator().manual_seed(seed)
        self.log = []

    @torch.no_grad()
    def prepare(self, obs):
        z = self.wm.target_encoder(obs)
        cells, _ = agent_cell(obs)
        flags = torch.stack(labels(obs), 1)
        return z, cells, flags

    def batch(self, prepared, batch, gen):
        return make_selection_batch(*prepared, self.cue, self.N_TOKENS, batch, gen)

    @torch.no_grad()
    def evaluate(self, prepared, batch=4000, seed=123):
        tokens, target, cell, _ = self.batch(prepared, batch, torch.Generator().manual_seed(seed))
        slots, _, attn = self.ws(tokens)
        cell_logits, _ = self.readout(slots)
        att = attn.mean(1)
        return {"attention": att[torch.arange(batch), target].mean().item(),
                "cell_acc": (cell_logits.argmax(1) == cell).float().mean().item()}

    def fit(self, train_obs, test_obs=None, eval_every=500, verbose=True):
        tr = self.prepare(train_obs)
        te = self.prepare(test_obs) if test_obs is not None else None
        for it in range(1, self.iters + 1):
            tokens, _, cell, flags = self.batch(tr, self.batch_size, self.gen)
            slots, _, _ = self.ws(tokens)
            cell_logits, flag_logits = self.readout(slots)
            loss = F.cross_entropy(cell_logits, cell) + \
                F.binary_cross_entropy_with_logits(flag_logits, flags)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if it % eval_every == 0 or it == 1:
                row = {"iter": it, "loss": loss.item()}
                if te is not None:
                    row.update(self.evaluate(te))
                self.log.append(row)
                if verbose and te is not None:
                    print(f"  [workspace] {it:5d}/{self.iters}  perte {row['loss']:.3f}  "
                          f"attention sur le signalé {row['attention']:.3f}  "
                          f"case retrouvée {100 * row['cell_acc']:.1f} %", flush=True)
        return self.log
