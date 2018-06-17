#! /usr/bin/env python3

import argparse
import json
import gzip

from players.model_player import ModelPlayer
from game.game import Game


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the model with Monte Carlo and a Deep Network")
    parser.add_argument("--infile", type=str, required=True,
                        help="Input file name.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    games = []
    with gzip.open(args.infile, 'r') as f: 
        for line in f:
            games.append(json.loads(line.decode()))

    model_player = ModelPlayer(10, 10, 10)

    player_1 = model_player
    player_2 = model_player
    player_3 = model_player

    score = {0: 0, 1:0, 2:0}
    for init_game in games:
        g = Game(init_game, [player_1, player_2, player_3])
        winners = g.play()
        for w in winners:
            score[w] += 1
        print(score)
