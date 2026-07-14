# Opta typeId / qualifierId constants used to identify set pieces.
#
# These match the MA1-style event feed used throughout this repo: each
# event has a `typeId` and a list of `qualifier` entries of the form
# `list(qualifierId = <int>, value = <...>)`.

TYPE_PASS <- 1L
TYPE_TACKLE <- 7L
TYPE_INTERCEPTION <- 8L
TYPE_CLEARANCE <- 12L
TYPE_MISS <- 13L
TYPE_POST <- 14L
TYPE_ATTEMPT_SAVED <- 15L
TYPE_GOAL <- 16L
TYPE_AERIAL <- 44L
TYPE_BALL_RECOVERY <- 49L

SHOT_TYPES <- c(TYPE_MISS, TYPE_POST, TYPE_ATTEMPT_SAVED, TYPE_GOAL)

SHOT_OUTCOME_NAMES <- c(
  `13` = "miss",
  `14` = "post",
  `15` = "saved",
  `16` = "goal"
)

# Contested actions used to find "the second ball" after a set-piece delivery
CONTESTED_TYPES <- c(TYPE_AERIAL, TYPE_TACKLE, TYPE_INTERCEPTION, TYPE_CLEARANCE, TYPE_BALL_RECOVERY)

QUALIFIER_FREE_KICK <- 5L
QUALIFIER_CORNER <- 6L
QUALIFIER_PENALTY <- 9L
QUALIFIER_THROW_IN <- 107L
QUALIFIER_PASS_END_X <- 140L
QUALIFIER_PASS_END_Y <- 141L
