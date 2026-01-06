import math

UNITID2CHAR = {
    0: ".",  # Empty space
    1: "F",  # Farmer
    2: "S",  # Assassin
    3: "K",  # TheKing
    4: "M",  # Mammoth
    5: "A",  # Archer
    6: "C",  # Cannon
    7: "D",  # Deadeye
    8: "H",  # Healer
    9: "P",  # Paladin
}
ALL_UNIT_NAMES = [
    "farmer",
    "assassin",
    "theking",
    "mammoth",
    "archer",
    "cannon",
    "deadeye",
    "healer",
    "paladin",
]

SIGHT_ANGLE = math.pi / 2
TURN_ANGLE = math.pi / 6


class UnitID:
    Farmer = 1  # F
    Assassin = 2  # S
    TheKing = 3  # K
    Mammoth = 4  # M
    Archer = 5  # A
    Cannon = 6  # C
    Deadeye = 7  # D
    Healer = 8  # H
    Paladin = 9  # P
