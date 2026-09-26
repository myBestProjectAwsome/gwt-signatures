"""Un épisode vécu : l'agent joue une tâche sur une carte, et on garde tout."""
import numpy as np

from ..data_keydoor import encode_events
from ..tasks import TaskTracker


def run_episode(agent, world, task, max_steps, on_step=None):
    """Renvoie (issue, nb de pas, trajectoire).

    trajectoire : dict de tableaux numpy (obs, action, next_obs, events), un par pas.
    on_step(world, décision, pas) : appelé après chaque action (affichage en direct).
    """
    tracker = TaskTracker(task)
    agent.reset(task)
    obs = world.observe()
    traj = {"obs": [], "action": [], "next_obs": [], "events": []}
    outcome = "bloqué"
    for t in range(1, max_steps + 1):
        action = agent.act(obs)
        nxt, events, dead = world.step(action)
        traj["obs"].append(obs.astype(np.uint8)); traj["action"].append(action)
        traj["next_obs"].append(nxt.astype(np.uint8)); traj["events"].append(encode_events(events))
        if on_step is not None:
            on_step(world, agent, t)
        obs = nxt
        if tracker.update(events):
            outcome = "succès"
            break
        if dead:
            outcome = "lave"
            break
    return outcome, t, {k: np.array(v) for k, v in traj.items()}
