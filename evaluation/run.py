"""Lancer le test.  python -m evaluation.run --references [--n 50]
                    python -m evaluation.run --test-final --je-confirme

--references : agents de référence (aléatoire, oracle) sur tous les cas.
               Sert à vérifier que la batterie fonctionne et que chaque carte est soluble.
--test-final : LE test (étape 5). Une seule fois. Déroulé, conforme à PROTOCOLE.md,
               sans rien modifier des règles figées dans battery.py :
  1. vérifications : protocole intact, isolement, résultats humains présents ;
  2. agent final (étape 4a + carte mentale), valeurs verrouillées (lock) ;
  3. zéro essai sur tous les cas (H1-H3, W1, W2, R0) ;
  4. adaptation : 50 épisodes par tâche nouvelle (cartes d'adaptation), puis mesure ;
  5. référence sans connaissances : même architecture, poids aléatoires, même adaptation ;
  6. R0 à nouveau (oubli), contrôle du verrou ;
  7. verdict() ; tout est écrit dans evaluation/resultat_final.json.
Si ce fichier existe déjà, le test a déjà eu lieu : le script refuse de le relancer.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mini_iag.config import Config

from .agents import OracleAgent, RandomAgent
from .battery import (N_MAPS, check_locked, practice, run_case, success_rate, summarize,
                      verdict)
from .isolation import check as isolation_problems
from .tasks import all_cases

HERE = Path(__file__).parent
ROOT = HERE.parent
FINAL = HERE / "resultat_final.json"
HUMAN = HERE / "human_results.json"


def integrity():
    """Compare les fichiers figés à l'empreinte enregistrée."""
    reg = HERE / "registration.json"
    if not reg.exists():
        return None
    data = json.loads(reg.read_text())
    ok = all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h
             for f, h in data["fichiers_figés"].items())
    return ok, data


def main():
    parser = argparse.ArgumentParser(prog="python -m evaluation.run")
    parser.add_argument("--references", action="store_true")
    parser.add_argument("--n", type=int, default=N_MAPS)
    parser.add_argument("--test-final", action="store_true")
    parser.add_argument("--je-confirme", action="store_true")
    args = parser.parse_args()
    state = integrity()
    if state is None:
        print("Protocole : pas encore enregistré (python -m evaluation.register).")
    else:
        print(f"Protocole enregistré le {state[1]['date']} : "
              + ("INTACT" if state[0] else "MODIFIÉ DEPUIS L'ENREGISTREMENT !"))
    if args.test_final:
        final_test(state, args)
        return
    if not args.references:
        parser.print_help()
        return
    cfg = Config()
    print(f"\n{'cas':5}{'tâche':22}{'cartes':12}" + "".join(f"{n:>22}" for n in ("aléatoire", "oracle")))
    for case in all_cases():
        row = f"{case.code:5}{case.task.name:22}{case.layout:12}"
        for agent in (RandomAgent(), OracleAgent()):
            s = summarize(run_case(agent, cfg, case, n=args.n))
            row += f"{s['succès']:9.1f} % lave {s['lave']:4.1f} %"
        print(row)


def final_test(state, args):
    import time

    import torch

    from mini_iag.final_agent import build_final_agent, build_relearner

    if FINAL.exists():
        print(f"\nLe test final a déjà eu lieu ({FINAL.name}). On ne le repasse pas.")
        return
    if state is None or not state[0]:
        print("\nProtocole absent ou modifié : test refusé.")
        return
    if not HUMAN.exists():
        print("\nRésultats humains absents : lancer d'abord python -m evaluation.human")
        return
    if args.n != N_MAPS:
        print(f"\nLe test final se passe sur {N_MAPS} cartes par cas (--n est ignoré).")
    if not args.je_confirme:
        print("\nLe test final ne se passe qu'UNE fois. Relancer avec --je-confirme.")
        return
    problems = isolation_problems()
    if problems:
        print("\nIsolement rompu : test refusé.\n  " + "\n  ".join(problems))
        return
    print("\nIsolement respecté.")
    t0, cfg, n = time.time(), Config(), N_MAPS
    torch.manual_seed(0)
    cases = all_cases()
    held = [c for c in cases if c.held_out_task]
    say = lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)

    agent = build_final_agent()
    say(f"Agent final chargé, valeurs verrouillées (empreinte {agent.cost_module.fingerprint()[:16]}...)")
    zero_shot = {}
    for c in cases:
        zero_shot[c.code] = run_case(agent, cfg, c, n)
        say(f"zéro essai  {c.code:3} {summarize(zero_shot[c.code])['succès']:5.1f} %")
    adapted = {}
    for c in held:
        practice(agent, cfg, c)
        adapted[c.code] = run_case(agent, cfg, c, n)
        say(f"adapté      {c.code:3} {summarize(adapted[c.code])['succès']:5.1f} %")
    r0 = next(c for c in cases if c.code == "R0")
    retention_after = success_rate(run_case(agent, cfg, r0, n))
    say(f"R0 après adaptations {100 * retention_after:5.1f} %")
    locked = check_locked(agent.cost_module)

    relearner, relearned = build_relearner(seed=0), {}
    for c in held:
        practice(relearner, cfg, c)
        relearned[c.code] = run_case(relearner, cfg, c, n)
        say(f"référence   {c.code:3} {summarize(relearned[c.code])['succès']:5.1f} %")

    human = json.loads(HUMAN.read_text())
    v = verdict(zero_shot, adapted, relearned, human, success_rate(zero_shot["R0"]),
                retention_after, locked)
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "protocole": state[1]["date"],
           "empreinte_valeurs": agent.cost_module.fingerprint(),
           "zéro_essai": {k: summarize(r) for k, r in zero_shot.items()},
           "adapté": {k: summarize(r) for k, r in adapted.items()},
           "référence_sans_connaissances": {k: summarize(r) for k, r in relearned.items()},
           "humain": {k: summarize(r) for k, r in human.items()},
           "verdict": v}
    FINAL.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print("\n=== VERDICT ===")
    for k in ("V1", "V2", "V3", "V4", "V5"):
        print(f"{k}  {'réussi' if v[k]['réussi'] else 'RATÉ ':7}  {v[k]['règle']}")
    print(f"\nmini-IAG : {'OUI' if v['mini-IAG'] else 'NON'}   (détails : {FINAL.name})")


if __name__ == "__main__":
    main()
