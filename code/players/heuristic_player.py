import requests
import sys
import json

class HeuristicPlayer:
    def __init__(self):
        self.type = 'heuristic'

    def choose_card(self, winner, game_type, current_pos, player_pos, cards,
                    current_trick, track, points):
        first_pos = 0
        game = {
            "name": game_type,
            "position": player_pos
        }
        payload = {
            "app": "Skat",
            "type": "bestCard",
            "game": game,
            "track": track,
            "cards": cards,
            "hand": True,
            "firstPosition": first_pos,
            "position": current_pos
        }
        # print(payload)
        return post_request(payload)


def post_request(payload):
    try:
        res = json.loads(
            requests.post(
                'http://127.0.0.1:10999/ai.json',
                json=payload).content.decode('utf-8')
        )
    except Exception as e:
        print(e)
        sys.exit(0)
    return None, res["cardID"], None, None
