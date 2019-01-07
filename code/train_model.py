#! /usr/bin/env python3

import yaml
import gzip
import json
import random

from code.players.model_player import ModelPlayer
from code.players.random_player import RandomPlayer
from code.players.heuristic_player import HeuristicPlayer
from code.game.game import Game
from code.model.memory import Memory


def main(job):
    games = []
    with gzip.open(job['GAMES_FILE'], 'r') as f:
        for line in f:
            games.append(json.loads(line.decode()))

    memory = Memory(job['MEMORY_SIZE'])
    model_player = ModelPlayer(job)
    heuristic_player = RandomPlayer()

    player_1 = model_player
    player_2 = model_player
    player_3 = model_player

    score = {0: 0, 1: 0, 2: 0}
    for init_game in games:
        if len(init_game['cards']) != 32:
            continue

        g = Game(memory, init_game, [player_1, player_2, player_3])
        winners = g.play()
        for w in winners:
            score[w] += 1
        print(score, len(memory.ltmemory))
        if len(memory.ltmemory) >= job['MEMORY_SIZE']:
            model_player.replay(memory.ltmemory, job)

            test_score = {0: 0, 1: 0, 2: 0}
            # for game in random.sample(games, 10):
            for game in games[:2]:
                if len(game['cards']) != 32:
                    continue
                g = Game(None, game, [player_1, heuristic_player, heuristic_player])
                # import ipdb; ipdb.set_trace()
                winners = g.play()
                for w in winners:
                    test_score[w] += 1
                print('test:', test_score)


def debug():
    filename = '/home/heiko/Projects/heiko/skat-zero/code/configuration.yml'
    with open(filename) as f:
        job = yaml.load(f)
    main(job)
