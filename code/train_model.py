#! /usr/bin/env python3

import yaml
import gzip
import json

from code.players.model_player import ModelPlayer
from code.game.game import Game


def main(job):
    games = []
    with gzip.open(job['GAMES_FILE'], 'r') as f:
        for line in f:
            games.append(json.loads(line.decode()))

    model_player = ModelPlayer(10, 10, 10, job)

    player_1 = model_player
    player_2 = model_player
    player_3 = model_player

    score = {0: 0, 1: 0, 2: 0}
    for init_game in games:
        g = Game(init_game, [player_1, player_2, player_3])
        winners = g.play()
        for w in winners:
            score[w] += 1
        print(score)


def debug():
    filename = '/home/heiko/Projects/heiko/skat-zero/code/configuration.yml'
    with open(filename) as f:
        job = yaml.load(f)
    main(job)
