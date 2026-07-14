"""Opta typeId / qualifierId constants used to identify set pieces.

These match the MA1-style event feed used throughout this repo: each
event has a ``typeId`` and a list of ``qualifier`` dicts of the form
``{"qualifierId": int, "value": ...}``.
"""

# Event typeId
TYPE_PASS = 1
TYPE_TACKLE = 7
TYPE_INTERCEPTION = 8
TYPE_CLEARANCE = 12
TYPE_MISS = 13
TYPE_POST = 14
TYPE_ATTEMPT_SAVED = 15
TYPE_GOAL = 16
TYPE_AERIAL = 44
TYPE_BALL_RECOVERY = 49

SHOT_TYPES = {TYPE_MISS, TYPE_POST, TYPE_ATTEMPT_SAVED, TYPE_GOAL}
SHOT_OUTCOME_NAMES = {
    TYPE_MISS: "miss",
    TYPE_POST: "post",
    TYPE_ATTEMPT_SAVED: "saved",
    TYPE_GOAL: "goal",
}

# Contested actions used to find "the second ball" after a set-piece delivery
CONTESTED_TYPES = {TYPE_AERIAL, TYPE_TACKLE, TYPE_INTERCEPTION, TYPE_CLEARANCE, TYPE_BALL_RECOVERY}

# Qualifiers
QUALIFIER_FREE_KICK = 5
QUALIFIER_CORNER = 6
QUALIFIER_PENALTY = 9
QUALIFIER_THROW_IN = 107
QUALIFIER_PASS_END_X = 140
QUALIFIER_PASS_END_Y = 141
