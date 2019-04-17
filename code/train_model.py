#! /usr/bin/env python3

import copy
import yaml
import gzip
import json
import random
import os
import logging

from code.players.model_player import ModelPlayer
from code.players.random_player import RandomPlayer
from code.players.heuristic_player import HeuristicPlayer
from code.game.game import Game
from code.model.memory import Memory
from code.funcs.play_matches import play_training_games, play_test_games, read_initial_games


def main(job):
    if not os.path.exists(job["LOG_DIR"]):
        os.makedirs(job["LOG_DIR"])
    if not os.path.exists(job["MODEL_DIR"]):
        os.makedirs(job["MODEL_DIR"])
    if not os.path.exists(job["RESULTS_DIR"]):
        os.makedirs(job["RESULTS_DIR"])

    logfile = os.path.join(job["LOG_DIR"], '{}.log'.format(job["JOB_ID"]))
    logging.basicConfig(filename=logfile, level=logging.DEBUG)
    resultsfile = os.path.join(job["RESULTS_DIR"], '{}.csv'.format(job["JOB_ID"]))
    rf = open(resultsfile, 'w+')

    memory = Memory(job['MEMORY_SIZE'])
    current_player = ModelPlayer(job)
    best_player = ModelPlayer(job)
    random_player = RandomPlayer()
    best_player_version = 0

    score = [0, 0, 0]
    test_score = [0, 0, 0]
    perform_score = [0, 0, 0]

    train_games, test_games = read_initial_games(job)

    for step in range(job['TRAINING_LOOPS']):
        players = [current_player, current_player, current_player]
        play_training_games(job, train_games, score, players, memory)
        rf.write('train,{},{},{}\n'.format(score[0], score[1], score[2]))
        print('train', score)

        players = [current_player, random_player, current_player]
        play_test_games(job, test_games, test_score, players)
        rf.write('test,{},{},{}\n'.format(test_score[0], test_score[1], test_score[2]))
        print('test', test_score)

        # players = [current_player, current_player, best_player]
        # play_test_games(job, test_games, perform_score, players)
        # rf.write('perform,{},{},{}\n'.format(perform_score[0], perform_score[1], perform_score[2]))
        # print('perform', perform_score)

        # if (test_score[0] + test_score[1]) / 2 >= test_score[2] * job['SCORING_THRESHOLD']:
        #     best_player_version = best_player_version + 1
        #     best_player.model.model.set_weights(current_player.model.model.get_weights())
        #     best_player.model.write(
        #         os.path.join(job["MODEL_DIR"], 'best_model_{}'.format(best_player_version))
        #     )
        # current_player.model.model.set_weights(best_player.model.model.get_weights())

        # import ipdb; ipdb.set_trace()


def debug():
    filename = '/home/heiko/Projects/heiko/skat-zero/code/configuration.yml'
    # filename = '/Users/heikoschmidle/projects/sauspiel/skat-zero/code/configuration.yml'
    # filename = '/src/configuration.yml'
    with open(filename) as f:
        job = yaml.load(f)
    main(job)
