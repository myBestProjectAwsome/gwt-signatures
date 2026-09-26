"""Module de coût configurable (monde v2) : le "configurateur" de LeCun (2022).

Le module évalue un état latent en estimant la probabilité de chaque événement
du monde : objectif, clé, porte, lave. La TÂCHE en cours (transmise par le
configurateur) choisit lequel compte comme succès :
    danger = P(lave)          succès = P(événement visé)
    coût   = w_danger x danger - w_succès x succès

Le même module sert donc à toutes les tâches, sans réapprentissage : changer de
tâche, c'est changer d'événement visé, pas de connaissances.

Monde v2 : le module juge une TRANSITION (état avant, état après), fournie
comme un seul vecteur [z_avant, z_après]. « Clé ramassée » ou « porte ouverte »
ne se voient pas sur l'état d'arrivée seul (un agent qui porte la clé depuis
dix pas lui ressemble), mais se voient sur le changement.

Les poids w_danger et w_succès sont des TAMPONS (buffers) : ils font partie de
l'état du module, donc de son empreinte SHA-256. Verrouiller le module verrouille
aussi les valeurs de l'agent (étape 5).
"""
import torch

from .cost_module import CostModule

EVENT_INDEX = {"goal": 0, "key": 1, "door": 2, "lava": 3}


class ConfigurableCost(CostModule):
    def __init__(self, cfg, w_danger=1.0, w_success=1.0):
        super().__init__(cfg)
        self.pair_input = cfg.pair_events
        in_dim = cfg.latent_dim * (2 if self.pair_input else 1)
        self.body[0] = torch.nn.Linear(in_dim, cfg.cost_hidden)
        self.body[-1] = torch.nn.Linear(cfg.cost_hidden, len(EVENT_INDEX))
        del self.w_danger, self.w_success
        self.register_buffer("w_danger", torch.tensor(float(w_danger)))
        self.register_buffer("w_success", torch.tensor(float(w_success)))
        self.target = "goal"

    def configure(self, event):
        """Choisit l'événement qui compte comme succès pour la tâche en cours."""
        if event not in EVENT_INDEX or event == "lava":
            raise ValueError(f"événement visé invalide : {event}")
        self.target = event

    def events(self, x):
        """Probabilité de chaque événement (B, 4) : objectif, clé, porte, lave.
        x : (B, 2D) transitions [z_avant, z_après] (ou (B, D) états si pair_input=False)."""
        return torch.sigmoid(self.body(x))

    def evaluate(self, z_prev, z_next):
        """(danger, succès, coût) d'une transition."""
        x = torch.cat([z_prev, z_next], dim=-1) if self.pair_input else z_next
        return self(x)

    def forward(self, z):
        p = self.events(z)
        danger, success = p[:, EVENT_INDEX["lava"]], p[:, EVENT_INDEX[self.target]]
        return danger, success, self.w_danger * danger - self.w_success * success
