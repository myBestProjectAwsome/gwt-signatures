"""Point d'entrée : python -m psyche [--db FICHIER] [--no-memory]"""
import argparse

from .loop import run


def main():
    parser = argparse.ArgumentParser(prog="python -m psyche")
    parser.add_argument("--db", default="data/episodes.sqlite",
                        help="fichier de mémoire (défaut : data/episodes.sqlite)")
    parser.add_argument("--no-memory", action="store_true",
                        help="lancer sans mémoire épisodique")
    parser.add_argument("--period", type=float, default=1.0,
                        help="durée d'un cycle en secondes (défaut : 1.0)")
    args = parser.parse_args()
    run(period=args.period, db_path=args.db, memory=not args.no_memory)


if __name__ == "__main__":
    main()
