class TaskTracker:
    """Suit la progression d'une tâche pendant un épisode."""

    def __init__(self, task):
        self.task = task
        self.progress = 0          # nb d'événements de la séquence déjà accomplis

    @property
    def done(self):
        return self.progress == len(self.task.sequence)

    @property
    def next_event(self):
        return None if self.done else self.task.sequence[self.progress]

    def update(self, events):
        """Prend en compte les événements d'un pas. Renvoie True si la tâche vient
        d'être réussie. Plusieurs événements du même pas sont traités dans l'ordre."""
        for e in events:
            if not self.done and e == self.next_event:
                self.progress += 1
        return self.done
