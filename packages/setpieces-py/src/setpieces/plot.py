"""Optional plotting helpers. Requires matplotlib:

    pip install "setpieces[plot]"
"""

from typing import Dict, Iterable, Optional

from .secondballs import SecondBallContest
from .zones import DEFAULT_ZONE_PARAMS, ZoneParams

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "setpieces.plot requires matplotlib. Install it with: pip install \"setpieces[plot]\""
    ) from exc


def _draw_pitch(ax, x0=0, x1=100, y0=0, y1=100, line_color="#3a4658", bg="#11161f"):
    ax.set_facecolor(bg)
    ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, edgecolor=line_color, linewidth=1.2))
    # 18-yard and 6-yard boxes on the attacking (high-x) end
    ax.add_patch(Rectangle((21, 83), 58, 17, fill=False, edgecolor=line_color, linewidth=1.0))
    ax.add_patch(Rectangle((37, 94.2), 26, 5.8, fill=False, edgecolor=line_color, linewidth=1.0))
    ax.set_xticks([])
    ax.set_yticks([])


def plot_zone_grid(zone_pct: Dict, ax=None, params: ZoneParams = DEFAULT_ZONE_PARAMS,
                    cmap: str = "viridis", title: Optional[str] = None):
    """Attacking-third pitch with the 6-zone delivery grid coloured by
    percentage. ``zone_pct`` is the output of
    ``setpieces.zones.zone_percentages`` (the "short" key, if present,
    is ignored -- it has no location to draw)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    _draw_pitch(ax, x0=50, x1=100)

    six_front = params.six_yard_front_x
    zones = [
        ("near", "edge", params.near_cut - 0, 0, params.box_front_x, six_front - params.box_front_x),
        ("central", "edge", params.far_cut - params.near_cut, params.near_cut, params.box_front_x, six_front - params.box_front_x),
        ("far", "edge", 100 - params.far_cut, params.far_cut, params.box_front_x, six_front - params.box_front_x),
        ("near", "six", params.near_cut - 0, 0, six_front, 100 - six_front),
        ("central", "six", params.far_cut - params.near_cut, params.near_cut, six_front, 100 - six_front),
        ("far", "six", 100 - params.far_cut, params.far_cut, six_front, 100 - six_front),
    ]
    cmap_fn = plt.get_cmap(cmap)
    vmax = max([v for k, v in zone_pct.items() if k != "short"], default=1.0)

    for col, row, width, y0, x0, height in zones:
        val = zone_pct.get((col, row), 0.0)
        color = cmap_fn(min(1.0, val / vmax)) if vmax else cmap_fn(0.0)
        ax.add_patch(Rectangle((y0, x0), width, height, facecolor=color, edgecolor="#11161f", linewidth=1.0))
        ax.text(y0 + width / 2, x0 + height / 2, f"{val:.0f}%", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(50, 100)
    if title:
        ax.set_title(title)
    return ax


def plot_second_ball_map(contests: Iterable[SecondBallContest], ax=None, title: Optional[str] = None):
    """Full pitch scatter of second-ball contests, coloured by whether
    the delivering team won the loose ball."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 10))

    _draw_pitch(ax, x0=0, x1=100)
    for c in contests:
        color = "#ffc247" if c.won else "#9aa4b2"
        alpha = 0.85 if c.won else 0.4
        ax.scatter(c.y, c.x, s=80, color=color, edgecolors="#11161f", linewidths=1.0, alpha=alpha, zorder=3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    if title:
        ax.set_title(title)
    return ax
