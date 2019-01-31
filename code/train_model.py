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
    current_player = ModelPlayer(job)
    best_player = ModelPlayer(job)
    heuristic_player = RandomPlayer()

    player_1 = current_player
    player_2 = current_player
    player_3 = current_player

    score = {0: 0, 1: 0, 2: 0}
    test_score = {0: 0, 1: 0, 2: 0}
    for step, init_game in enumerate(train_games):
        ig = copy.deepcopy(init_game)
        g = Game(memory, ig, [player_1, player_2, player_3])
        winners = g.play()
        if winners is None:
            continue
        for w in winners:
            score[w] += 1

        print(score)

        if len(memory.ltmemory) >= job['MEMORY_SIZE']:
            current_player.replay(memory.ltmemory, job)
            memory.clear_ltmemory()

        if step % 1 == 0:
            for init_game in random.sample(test_games, 3):
                tg = copy.deepcopy(init_game)
                g = Game(None, tg, [player_1, heuristic_player, heuristic_player])
                winners = g.play()
                if winners is None:
                    continue
                for w in winners:
                    test_score[w] += 1
            print('test:', test_score)
            import ipdb; ipdb.set_trace()
            player_1 = best_player
            for init_game in random.sample(test_games, 3):
                tg = copy.deepcopy(init_game)
                g = Game(None, tg, [player_1, heuristic_player, heuristic_player])
                winners = g.play()
                if winners is None:
                    continue
                for w in winners:
                    test_score[w] += 1
            print('test:', test_score)
            # import ipdb; ipdb.set_trace()


def debug():
    # filename = '/home/heiko/Projects/heiko/skat-zero/code/configuration.yml'
    filename = '/Users/heikoschmidle/projects/sauspiel/skat-zero/code/configuration.yml'
    with open(filename) as f:
        job = yaml.load(f)
    main(job)
