from torch import nn


class SelectionReadout(nn.Module):
    """Tête de lecture pour entraîner le workspace : elle ne voit QUE les slots.

    À partir des K slots du workspace, elle doit retrouver ce que contient le
    contenu signalé : la case de l'agent (49 classes), danger, succès.
    Comme elle ne voit pas les contenus directement, toute l'information doit
    passer par le goulot : le workspace doit apprendre à laisser entrer le bon.
    """

    def __init__(self, cfg):
        super().__init__()
        d, k, n = cfg.latent_dim, cfg.workspace_slots, cfg.grid_size
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(k * d, 128), nn.ReLU())
        self.cell = nn.Linear(128, n * n)
        self.flags = nn.Linear(128, 2)

    def forward(self, slots):
        h = self.net(slots)
        return self.cell(h), self.flags(h)
