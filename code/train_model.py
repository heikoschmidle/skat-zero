#! /usr/bin/env python3

import copy
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
    train_games = []
    test_games = []
    with gzip.open(job['GAMES_FILE'], 'r') as f:
        for line in f:
            game = json.loads(line.decode())
            if len(game['cards']) != 32:
                print(game)
                continue
            if random.random() < 0.7:
                train_games.append(game)
            else:
                test_games.append(game)

    memory = Memory(10000)  # job['MEMORY_SIZE'])
    model_player = ModelPlayer(job)
    heuristic_player = RandomPlayer()

    player_1 = model_player
    player_2 = model_player
    player_3 = model_player

    score = {0: 0, 1: 0, 2: 0}
    test_score = {0: 0, 1: 0, 2: 0}
    for init_game in train_games:
        # ig = copy.deepcopy(init_game)
        ig = init_game
        g = Game(memory, ig, [player_1, player_2, player_3])
        winners = g.play()
        for w in winners:
            score[w] += 1
        print(score, len(memory.ltmemory))
        if len(memory.ltmemory) >= job['MEMORY_SIZE']:
            model_player.replay(memory.ltmemory, job)
            memory.clear_ltmemory()

            for tg in random.sample(train_games, 3):
                # print(tg)
                # print(tg['cards'])
            # for game in games[:2]:
                # if len(tg['cards']) != 32:
                    # continue
            # for _ in range(5):
                # tg = {'cards': ['S7', 'SX', 'G7', 'S8', 'E9', 'SU', 'G8', 'HK', 'H9', 'GO', 'SA', 'HX', 'EA', 'GK', 'HU', 'SO', 'GA', 'GU', 'E8', 'HA', 'E7', 'EX', 'H8', 'SK', 'GX', 'HO', 'S9', 'H7', 'EO', 'G9', 'EK', 'EU'], 'player': {'reizwert': 48, 'name': 'grand', 'position': 1, 'play_hand': True}}
                # import ipdb; ipdb.set_trace()
                g = Game(None, tg, [player_1, heuristic_player, heuristic_player])
                # import ipdb; ipdb.set_trace()
                winners = g.play()
                for w in winners:
                    test_score[w] += 1
            print('test:', test_score)
            # import ipdb; ipdb.set_trace()


def debug():
    filename = '/home/heiko/Projects/heiko/skat-zero/code/configuration.yml'
    with open(filename) as f:
        job = yaml.load(f)
    main(job)
