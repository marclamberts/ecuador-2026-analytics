"""
Create a more experimental visual pack for headed-clearance outcome analysis.

These visuals are intentionally different from the standard pitch maps:
  - expected-to-actual displacement map
  - risk/relief landscape
  - chronological style barcode
  - decision compass
  - second-wave storyboard
  - league risk surface with player overlay

Example:
  python3 create_innovative_header_clearance_visuals.py --player "C. Gruezo"
"""

from __future__ import annotations

import argparse
import pathlib
import sys


LOCAL_PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1] / "Statsbomb" / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mplsoccer import Pitch


HERE = pathlib.Path(__file__).resolve().parent
OUTCOME_DIR = HERE / "ClearanceLandingModel" / "OutcomeModel"
PREDICTIONS_PATH = OUTCOME_DIR / "headed_clearance_outcome_predictions.csv"
OUT_DIR = OUTCOME_DIR / "innovative_visuals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create innovative headed-clearance visuals.")
    parser.add_argument("--player", default="C. Gruezo")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    return parser.parse_args()


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def make_pitch() -> Pitch:
    return Pitch(
        pitch_type="opta",
        pitch_color="#101820",
        line_color="#d7dde2",
        linewidth=1.15,
        goal_type="box",
    )


def dark_axes(ax) -> None:
    ax.set_facecolor("#101820")
    ax.tick_params(colors="#d7dde2")
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.grid(color="#405160", alpha=0.28, lw=0.7)


def value_cmap():
    return LinearSegmentedColormap.from_list("value", ["#ff4d6d", "#2d3340", "#7ee081"])


def save_displacement_map(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.2), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)

    vmax = max(abs(player["decision_quality_oe"].min()), abs(player["decision_quality_oe"].max()), 1)
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = value_cmap()

    for _, row in player.iterrows():
        color = cmap(norm(row["decision_quality_oe"]))
        pitch.scatter(row["pred_landing_x"], row["pred_landing_y"], ax=ax, s=58, c=["#7aa7ff"], alpha=0.62, edgecolors="none", zorder=3)
        pitch.scatter(row["landing_x"], row["landing_y"], ax=ax, s=92, c=[color], edgecolors="white", lw=0.7, zorder=5)
        pitch.arrows(
            row["pred_landing_x"],
            row["pred_landing_y"],
            row["landing_x"],
            row["landing_y"],
            ax=ax,
            color=color,
            width=1.8,
            headwidth=5,
            headlength=5,
            alpha=0.82,
            zorder=4,
        )

    ax.set_title(
        f"{player.iloc[0]['player_name']} | Beating the Expected Landing",
        color="#f8fafc",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        50,
        -7,
        "Blue dots are model-expected landings. Arrows show how actual headers moved away from expectation. Green = better decision outcome, red = worse.",
        ha="center",
        va="center",
        color="#9fb2c3",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_risk_relief_landscape(df: pd.DataFrame, player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 8.2), facecolor="#0b1117")
    dark_axes(ax)

    x = df["p_relief_success"]
    y = df["p_opponent_shot_10s"]
    ax.hexbin(x, y, C=df["decision_quality_oe"], gridsize=22, reduce_C_function=np.mean, cmap=value_cmap(), mincnt=2, alpha=0.78)
    sc = ax.scatter(
        player["p_relief_success"],
        player["p_opponent_shot_10s"],
        s=np.clip(player["clearance_length"] * 5.2, 70, 260),
        c=player["decision_quality_oe"],
        cmap=value_cmap(),
        edgecolors="white",
        lw=0.7,
        zorder=5,
    )
    ax.axvline(df["p_relief_success"].median(), color="#d7dde2", lw=1.0, ls="--", alpha=0.75)
    ax.axhline(df["p_opponent_shot_10s"].median(), color="#d7dde2", lw=1.0, ls="--", alpha=0.75)
    ax.text(0.93, 0.02, "High relief\nlow shot risk", color="#7ee081", fontsize=11, ha="right", va="bottom", transform=ax.transAxes)
    ax.text(0.07, 0.94, "Low relief\nhigh shot risk", color="#ff6b6b", fontsize=11, ha="left", va="top", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.45, df["p_opponent_shot_10s"].quantile(0.995)))
    ax.set_xlabel("Predicted relief success", color="#d7dde2")
    ax.set_ylabel("Predicted opponent shot within 10s", color="#d7dde2")
    ax.set_title(f"{player.iloc[0]['player_name']} | Risk-Relief Landscape", color="#f8fafc", fontsize=18, fontweight="bold", pad=12)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Decision quality OE", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_style_barcode(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    chart = player.sort_values(["match_id", "period_id", "elapsed_seconds"]).copy()
    chart["event_no"] = np.arange(len(chart))
    styles = list(chart["style_name"].dropna().unique())
    colors = {style: plt.get_cmap("tab10")(i % 10) for i, style in enumerate(styles)}

    fig, ax = plt.subplots(figsize=(13, 4.8), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    for _, row in chart.iterrows():
        color = colors.get(row["style_name"], "#9fb2c3")
        alpha = 1.0 if row["decision_quality_oe"] >= 0 else 0.42
        ax.add_patch(plt.Rectangle((row["event_no"], 0), 0.92, 1.0, color=color, alpha=alpha))
        if row["opponent_shot_10s"]:
            ax.plot(row["event_no"] + 0.46, 1.16, marker="X", color="#ff4d6d", ms=9, markeredgecolor="white", markeredgewidth=0.5)
        elif row["relief_success"]:
            ax.plot(row["event_no"] + 0.46, 1.13, marker="o", color="#7ee081", ms=7, markeredgecolor="white", markeredgewidth=0.45)

    ax.set_xlim(-0.2, len(chart) + 0.2)
    ax.set_ylim(-0.12, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("Headed clearance sequence", color="#d7dde2")
    ax.tick_params(axis="x", colors="#d7dde2")
    for spine in ax.spines.values():
        spine.set_color("#405160")
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[s], label=s) for s in styles]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, frameon=False, labelcolor="#f8fafc", fontsize=9)
    ax.set_title(
        f"{chart.iloc[0]['player_name']} | Style Barcode",
        color="#f8fafc",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    ax.text(0, -0.04, "Bright = positive decision quality. Dim = negative. Green dot = relief. Red X = opponent shot within 10s.", color="#9fb2c3", fontsize=9)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_decision_compass(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    metrics = {
        "Relief OE": player["relief_oe"].mean(),
        "Shot prevention": player["shot_prevention_oe"].mean(),
        "Length OE": player["length_oe"].mean() / 15.0,
        "Territory OE": player["territory_oe"].mean() / 10.0,
        "Wide OE": player["wide_oe"].mean() / 12.0,
        "Value OE": player["clearance_value_oe"].mean() / 8.0,
    }
    labels = list(metrics.keys())
    values = np.clip(np.array(list(metrics.values()), dtype=float), -1, 1)
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    width = 2 * np.pi / len(labels) * 0.72
    colors = [value_cmap()(Normalize(-1, 1)(v)) for v in values]

    fig = plt.figure(figsize=(8.6, 8.6), facecolor="#0b1117")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#101820")
    ax.bar(theta, np.abs(values), width=width, bottom=0, color=colors, alpha=0.88, edgecolor="#f8fafc", linewidth=0.8)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels, color="#f8fafc", fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25", ".50", ".75", "1.0"], color="#7d90a1", fontsize=8)
    ax.grid(color="#405160", alpha=0.55)
    ax.spines["polar"].set_color("#405160")
    ax.set_title(f"{player.iloc[0]['player_name']} | Decision Compass", color="#f8fafc", fontsize=18, fontweight="bold", pad=22)
    ax.text(0.5, -0.08, "Bars are scaled over/under expected components. Green points outward as positive.", transform=ax.transAxes, ha="center", color="#9fb2c3", fontsize=9)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_storyboard(player: pd.DataFrame, out_path: pathlib.Path, n: int = 6) -> None:
    sample = player.reindex(player["decision_quality_oe"].abs().sort_values(ascending=False).head(n).index)
    pitch = make_pitch()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), facecolor="#0b1117")
    axes = axes.flat
    for ax, (_, row) in zip(axes, sample.iterrows()):
        pitch.draw(ax=ax)
        pitch.scatter(row["start_x"], row["start_y"], ax=ax, s=80, c="#f25f5c", edgecolors="white", lw=0.6, zorder=4)
        pitch.scatter(row["pred_landing_x"], row["pred_landing_y"], ax=ax, s=65, c="#7aa7ff", edgecolors="none", alpha=0.75, zorder=3)
        pitch.scatter(row["landing_x"], row["landing_y"], ax=ax, s=95, c="#ffbf69", edgecolors="white", lw=0.65, zorder=5)
        pitch.lines(row["start_x"], row["start_y"], row["landing_x"], row["landing_y"], ax=ax, color="#ffbf69", lw=1.3, alpha=0.75)
        pitch.arrows(row["pred_landing_x"], row["pred_landing_y"], row["landing_x"], row["landing_y"], ax=ax, color="#f8fafc", width=1.3, headwidth=4, headlength=4, alpha=0.75)
        outcome = "RELIEF" if row["relief_success"] else "NO RELIEF"
        if row["opponent_shot_10s"]:
            outcome = "SHOT RISK HIT"
        title = f"{outcome} | DQ {row['decision_quality_oe']:+.1f}"
        ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold")
    for ax in axes[len(sample) :]:
        ax.axis("off")
    fig.suptitle(f"{player.iloc[0]['player_name']} | Six Clearance Storyboards", color="#f8fafc", fontsize=19, fontweight="bold")
    fig.text(0.5, 0.03, "Red = origin, blue = expected landing, amber = actual landing, white arrow = expectation displacement.", ha="center", color="#9fb2c3", fontsize=10)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_league_risk_surface(df: pd.DataFrame, player: pd.DataFrame, out_path: pathlib.Path) -> None:
    pitch = make_pitch()
    fig, ax = plt.subplots(figsize=(11.5, 8.2), facecolor="#0b1117")
    pitch.draw(ax=ax)
    bin_stat = pitch.bin_statistic(df["landing_x"], df["landing_y"], values=df["p_opponent_shot_10s"], statistic="mean", bins=(16, 12))
    pitch.heatmap(bin_stat, ax=ax, cmap="magma_r", edgecolors="#101820", alpha=0.88)
    pitch.scatter(player["landing_x"], player["landing_y"], ax=ax, s=110, c="#7ee081", edgecolors="white", lw=0.7, zorder=5)
    pitch.scatter(player["pred_landing_x"], player["pred_landing_y"], ax=ax, s=58, c="#7aa7ff", edgecolors="none", alpha=0.72, zorder=4)
    ax.set_title(
        f"{player.iloc[0]['player_name']} on League Shot-Risk Surface",
        color="#f8fafc",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        50,
        -7,
        "Background = average predicted opponent shot risk by landing zone. Green = actual Gruezo landings. Blue = expected landings.",
        ha="center",
        color="#9fb2c3",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PREDICTIONS_PATH)
    player = df[df["player_name"].str.casefold() == args.player.casefold()].copy()
    if player.empty:
        raise SystemExit(f"No headed-clearance outcome rows found for {args.player}")

    prefix = safe_name(player.iloc[0]["player_name"])
    paths = [
        args.out_dir / f"{prefix}_expectation_displacement_map.png",
        args.out_dir / f"{prefix}_risk_relief_landscape.png",
        args.out_dir / f"{prefix}_style_barcode.png",
        args.out_dir / f"{prefix}_decision_compass.png",
        args.out_dir / f"{prefix}_clearance_storyboard.png",
        args.out_dir / f"{prefix}_league_risk_surface.png",
    ]
    save_displacement_map(player, paths[0])
    save_risk_relief_landscape(df, player, paths[1])
    save_style_barcode(player, paths[2])
    save_decision_compass(player, paths[3])
    save_storyboard(player, paths[4])
    save_league_risk_surface(df, player, paths[5])

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
