import copy
import random
import gzip
import json

from code.game.game import Game


def read_initial_games(job):
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
    return train_games, test_games


def play_training_games(job, train_games, score, players, memory):
    # import ipdb; ipdb.set_trace()
    for idx, init_game in enumerate(random.sample(train_games, job['TRAINING_SIZE'])):
        print("Training {} of {}".format(idx, job['TRAINING_SIZE']))
        ig = copy.deepcopy(init_game)
        g = Game(memory, ig, players)
        winners = g.play()
        if winners is None:
            continue
        for w in winners:
            score[w] += 1

        if len(memory.ltmemory) >= job['MEMORY_SIZE']:
            players[0].replay(memory.ltmemory, job)
            memory.clear_ltmemory()
    # import ipdb; ipdb.set_trace()


def play_test_games(job, test_games, test_score, players):
    for init_game in random.sample(test_games, job['TEST_SIZE']):
        tg = copy.deepcopy(init_game)
        g = Game(None, tg, players)
        winners = g.play()
        if winners is None:
            print("No winner???")
            continue
        for w in winners:
            test_score[w] += 1
