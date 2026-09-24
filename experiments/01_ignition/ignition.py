"""
ignition.py — Expérience 1 : recherche d'une ignition globale (Dehaene & Changeux).

Modèle : N modules (intégrateurs à fuite) + 1 noeud Global Workspace (GW).
 - Le module 0 ("capteur") reçoit un stimulus bref d'intensité variable.
 - Les modules envoient leur activité au GW (feed-forward).
 - Le GW s'auto-entretient (réverbération) et renvoie son activité à TOUS
   les modules (broadcast).
Aucun comportement "conscient" n'est écrit en dur : si une ignition
apparaît, elle émerge de la dynamique.

Prédictions GWT testées :
 P1  Avec broadcast, l'accès au GW est non linéaire (seuil net en intensité).
 P2  Près du seuil, les essais sont BIMODAUX : tout ou rien, pas d'intermédiaire
     (cf. Sergent & Dehaene 2004, clignement attentionnel).
 P3  Après ignition, des modules NON stimulés sont recrutés (broadcast global).
 P4  Ablation (pas de réverbération ni de broadcast) : réponse graduelle,
     locale et transitoire. P1-P3 disparaissent.
"""
import numpy as np

PARAMS = dict(n=8, steps=300, dt=0.1, stim_dur=30, noise=0.08,
              w_self=0.3, w_ff=1.2, w_gg=0.55, w_fb=0.45, th_gw=0.8)


def sigmoid(x, gain=8.0, th=0.5):
    return 1 / (1 + np.exp(-gain * (x - th)))


def trial(intensity, broadcast=True, rng=None, p=PARAMS):
    rng = rng or np.random.default_rng()
    n, steps, dt = p["n"], p["steps"], p["dt"]
    w_gg = p["w_gg"] if broadcast else 0.0
    w_fb = p["w_fb"] if broadcast else 0.0
    x, g = np.zeros(n), 0.0
    g_tr, x_tr = np.zeros(steps), np.zeros((steps, n))
    for t in range(steps):
        inp = np.zeros(n)
        if t < p["stim_dur"]:
            inp[0] = intensity
        x += dt * (-x + sigmoid(p["w_self"] * x + w_fb * g + inp
                                + p["noise"] * rng.standard_normal(n)))
        ff = p["w_ff"] * (0.6 * x.max() + 0.4 * x.mean())
        g += dt * (-g + sigmoid(ff + w_gg * g + p["noise"] * rng.standard_normal(),
                                th=p["th_gw"]))
        g_tr[t], x_tr[t] = g, x
    late = slice(p["stim_dur"] + 100, steps)
    return dict(g_peak=g_tr.max(), g_late=g_tr[late].mean(),
                recruited=x_tr[late, 1:].mean(), g_tr=g_tr, x_tr=x_tr)


def sweep(broadcast, intensities, n_trials, rng):
    res = {"g_peak": [], "p_ign": [], "recruited": [], "g_late_all": []}
    for s in intensities:
        r = [trial(s, broadcast, rng) for _ in range(n_trials)]
        gl = np.array([q["g_late"] for q in r])
        res["g_peak"].append(np.mean([q["g_peak"] for q in r]))
        res["p_ign"].append(np.mean(gl > 0.5))
        res["recruited"].append(np.mean([q["recruited"] for q in r]))
        res["g_late_all"].append(gl)
    return res


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    I = np.linspace(0, 1.2, 25)
    N = 60
    on = sweep(True, I, N, rng)
    off = sweep(False, I, N, rng)

    # Intensité la plus proche du seuil (p_ign ~ 0.5)
    k = int(np.argmin(np.abs(np.array(on["p_ign"]) - 0.5)))
    s_th = I[k]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.plot(I, on["g_peak"], "o-", label="avec broadcast")
    a.plot(I, off["g_peak"], "s--", label="ablation")
    a.set(title="P1/P4 — Activité max du workspace", xlabel="intensité du stimulus",
          ylabel="pic de g")
    a.legend()

    a = ax[0, 1]
    a.plot(I, on["recruited"], "o-", label="avec broadcast")
    a.plot(I, off["recruited"], "s--", label="ablation")
    a.set(title="P3 — Recrutement des modules non stimulés (tardif)",
          xlabel="intensité du stimulus", ylabel="activité moyenne modules 1..N")
    a.legend()

    a = ax[1, 0]
    a.hist(on["g_late_all"][k], bins=20, range=(0, 1), alpha=0.7, label="avec broadcast")
    a.hist(off["g_late_all"][k], bins=20, range=(0, 1), alpha=0.7, label="ablation")
    a.set(title=f"P2 — Distribution des essais au seuil (s={s_th:.2f})",
          xlabel="activité tardive du workspace", ylabel="nb d'essais")
    a.legend()

    a = ax[1, 1]
    rng2 = np.random.default_rng(7)
    shown = {True: 0, False: 0}
    for _ in range(40):
        r = trial(s_th, True, rng2)
        ign = r["g_late"] > 0.5
        if shown[ign] < 4:
            a.plot(r["g_tr"], color="tab:red" if ign else "tab:blue", alpha=0.8)
            shown[ign] += 1
    a.axvspan(0, PARAMS["stim_dur"], color="grey", alpha=0.2, label="stimulus")
    a.set(title="Essais individuels au seuil : ignition (rouge) vs échec (bleu)",
          xlabel="pas de temps", ylabel="g")
    a.legend()

    fig.tight_layout()
    fig.savefig("ignition_results.png", dpi=120)

    print(f"Seuil estimé : s ≈ {s_th:.2f}")
    print(f"Essais au seuil (broadcast) : {np.mean(on['g_late_all'][k] > 0.5):.0%} ignitions, "
          f"{np.mean((on['g_late_all'][k] > 0.2) & (on['g_late_all'][k] < 0.5)):.0%} intermédiaires")
    print(f"Ablation, intensité max : pic g = {off['g_peak'][-1]:.2f}, "
          f"g tardif = {off['g_late_all'][-1].mean():.2f}")
