#! /usr/bin/env python3

import argparse
import gzip
import json

from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer
from game.initial_game import Game


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create initial game setups")
    parser.add_argument("--number", type=int, required=False, default=1000,
                        help="How many games to create?")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Output file name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n = 0
    games = []
    while True: 
        print(n)
        game = Game()
        if len(game.player) == 0:
            continue
        games.append(game.game)
        n += 1
        if n == args.number:
            break

    print("Writing...")
    with gzip.open(args.outfile, 'w') as f: 
        for g in games:
            f.write((json.dumps(g) + "\n").encode())
