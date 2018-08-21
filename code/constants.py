CARDS = [
    "EU",
    "GU",
    "HU",
    "SU",
    "EA",
    "EX",
    "EK",
    "EO",
    "E9",
    "E8",
    "E7",
    "GA",
    "GX",
    "GK",
    "GO",
    "G9",
    "G8",
    "G7",
    "HA",
    "HX",
    "HK",
    "HO",
    "H9",
    "H8",
    "H7",
    "SA",
    "SX",
    "SK",
    "SO",
    "S9",
    "S8",
    "S7"
]

NULL_CARDS = [
    "EA",
    "EK",
    "EO",
    "EU",
    "EX",
    "E9",
    "E8",
    "E7",
    "GA",
    "GK",
    "GO",
    "GU",
    "GX",
    "G9",
    "G8",
    "G7",
    "HA",
    "HK",
    "HO",
    "HU",
    "HX",
    "H9",
    "H8",
    "H7",
    "SA",
    "SK",
    "SO",
    "SU",
    "SX",
    "S9",
    "S8",
    "S7"
]

SUIT_MAP = {
    "kreuz": "E",
    "pik": "G",
    "herz": "H",
    "karo": "S"
}

POINTS = {
    "A": 11,
    "X": 10,
    "K": 4,
    "O": 3,
    "U": 2
}

def sort_cards(cards):
    sorted_cards = []
    for c in CARDS:
        if c in cards:
            sorted_cards.append(c)
    return sorted_cards

def encode_binary(cards):
    res = [0] * 32
    for c in cards:
        res[CARDS.index(c)] = 1
    return res
