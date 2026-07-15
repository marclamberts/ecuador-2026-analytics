"""Shared team color identity for the PossessionMDP visuals, so the same
team reads as the same color across every chart in this folder."""
from __future__ import annotations

TEAM_COLORS = [
    "#4da3e8", "#ff9179", "#f2c14e", "#9b8ce0", "#57c785", "#e069a6",
    "#5ec8d8", "#e0904a", "#a8d65e", "#c179d1", "#e0555f", "#4fb8a0",
    "#f0a13c", "#7b93e0", "#c9a13c", "#6fd1a0",
]


def team_color_map(team_names: list[str]) -> dict[str, str]:
    """Stable color per team name, ordered alphabetically so the mapping
    doesn't shuffle if the input order changes between scripts/runs."""
    return {name: TEAM_COLORS[i % len(TEAM_COLORS)] for i, name in enumerate(sorted(team_names))}
