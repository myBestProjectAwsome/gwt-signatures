from mini_iag.tasks import oracle_action


class OracleAgent:
    """Référence haute : plus court chemin exact. TRICHE (lit l'état du monde) :
    sert seulement à vérifier que chaque carte est soluble et à borner les scores."""
    name = "oracle"
    privileged = True

    def reset(self, task):
        self.task = task

    def act(self, obs, world=None, progress=0):
        return oracle_action(world, self.task, progress)
