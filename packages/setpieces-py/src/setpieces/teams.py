"""Convenience helper to map team names to Opta contestantIds from this
repo's match-file naming convention: ``YYYY-MM-DD_Home Team - Away Team.json``.

This is optional -- every other function in the package works directly
off ``contestantId`` strings, so use this only if your file layout
follows that convention.
"""

import collections
import json
import re
from typing import Dict, List

FILENAME_RE = re.compile(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$")


def team_ids_from_filenames(paths: List[str]) -> Dict[str, str]:
    """team display name -> contestantId, inferred by intersecting the
    contestantIds seen in every match file where that team's name
    appears as the home or away side."""
    team_cid_sets = collections.defaultdict(list)
    for path in paths:
        m = FILENAME_RE.search(path.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cids = {e["contestantId"] for e in data.get("event", []) if "contestantId" in e}
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)

    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid
