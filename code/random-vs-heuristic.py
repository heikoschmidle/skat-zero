#! /usr/bin/env python3

import argparse
import gzip
import json

from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer
from game.game import Game


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play different players against each other.")
    parser.add_argument("--infile", type=str, required=True,
                        help="Input file name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    games = []
    with gzip.open(args.infile, 'r') as f: 
        for line in f:
            games.append(json.loads(line.decode()))

    player_1 = HeuristicPlayer()
    player_2 = HeuristicPlayer()
    player_3 = RandomPlayer()

    score = {0: 0, 1:0, 2:0}
    for init_game in games:
        g = Game(init_game, [player_1, player_2, player_3])
        winners = g.play()
        for w in winners:
            score[w] += 1
        print(score)
