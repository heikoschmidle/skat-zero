def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory=None, goes_first=0):
    env = Game()
    scores = {
        player1.name: 0,
        "drawn": 0,
        player2.name: 0
    }
    sp_scores = {
        'sp': 0,
        "drawn": 0,
        'nsp': 0
    }
    points = {
        player1.name: [],
        player2.name: []
    }

    for e in range(EPISODES):
        state = env.reset()
        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if goes_first == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {
                1: {
                    "agent": player1,
                    "name": player1.name
                },
                -1: {
                    "agent": player2,
                    "name": player2.name
                }
            }
        else:
            players = {
                1: {
                    "agent": player2,
                    "name": player2.name
                },
                -1: {
                    "agent": player1,
                    "name": player1.name
                }
            }

        while done == 0:
            turn = turn + 1
            # Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)

            if memory is not None:
                # Commit the move to memory
                memory.commit_stmemory(env.identities, state, pi)

            # Do the action
            state, value, done, _ = env.step(action)
            if done == 1:
                if memory is not None:
                    # If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value
                    memory.commit_ltmemory()
                if value == 1:
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])

    return (scores, memory, points, sp_scores)
