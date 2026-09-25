from torch import nn


class EventPredictor(nn.Module):
    """Latent -> logits des événements perçus dans cet état (lave, objectif).

    Tête auxiliaire du modèle du monde. Sans elle, le JEPA JETTE la position
    de l'objectif : l'objectif ne change rien aux déplacements, donc il est
    inutile pour prédire l'état suivant (cf. README, étape 2). Or le module de
    coût en a besoin. En demandant au modèle du monde de prédire aussi ce qu'il
    va PERCEVOIR, on l'oblige à garder ce qui compte pour l'agent.

    Rôles distincts : le modèle du monde prédit CE QUI VA ARRIVER ; le module
    de coût décide COMBIEN C'EST MAUVAIS (et sera verrouillé à l'étape 5).
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.predictor_hidden), nn.ReLU(),
            nn.Linear(cfg.predictor_hidden, 2),
        )

    def forward(self, z):
        return self.net(z)
