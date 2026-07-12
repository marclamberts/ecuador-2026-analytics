"""
Build analytical outputs and visuals for headed-clearance landing quality.

This script adds the analysis layer on top of the landing model:
  - per-clearance over-expectation metrics
  - next-action outcomes after the clearance
  - player and team summaries
  - player visuals for tactical analysis

Example:
  python3 build_header_clearance_insights.py --player "C. Gruezo"
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys


LOCAL_PACKAGE_DIR = pathlib.Path(__file__).resolve().parents[1] / "Statsbomb" / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch


HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = HERE / "ClearanceLandingModel"
MODEL_PATH = MODEL_DIR / "headed_clearance_landing_model.joblib"
DATASET_PATH = MODEL_DIR / "headed_clearances_dataset.csv"
OUT_DIR = MODEL_DIR / "Insights"
VIS_DIR = OUT_DIR / "player_visuals"

SHOT_TYPES = {13, 14, 15, 16}
ADMIN_TYPES = {18, 19, 30, 32, 34, 37, 40, 70, 71, 90, 91}
TOUCH_EXCLUDE_TYPES = ADMIN_TYPES | {27, 28}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create headed-clearance insight outputs.")
    parser.add_argument("--player", default="C. Gruezo")
    parser.add_argument("--data-dir", type=pathlib.Path, default=HERE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    return parser.parse_args()


def make_pitch() -> Pitch:
    return Pitch(
        pitch_type="opta",
        pitch_color="#101820",
        line_color="#d7dde2",
        linewidth=1.2,
        goal_type="box",
    )


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def elapsed_seconds(event: dict) -> float:
    period = int(event.get("periodId") or 0)
    base = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}.get(period, 0)
    return base + int(event.get("timeMin") or 0) * 60 + int(event.get("timeSec") or 0)


def landing_zone(x: float, y: float) -> str:
    x_zone = "defensive" if x < 33.333 else "middle" if x < 66.667 else "attacking"
    y_zone = "left" if y < 33.333 else "central" if y < 66.667 else "right"
    return f"{x_zone}_{y_zone}"


def is_danger_zone(x: float, y: float) -> bool:
    return x < 40 and 33.333 <= y <= 66.667


def is_safe_wide_zone(x: float, y: float) -> bool:
    return x >= 30 and (y < 25 or y > 75)


def load_match_events(data_dir: pathlib.Path) -> dict[str, list[dict]]:
    events_by_match = {}
    for path in sorted(data_dir.glob("*.json")):
        with path.open() as f:
            events = json.load(f).get("event", [])
        events.sort(key=lambda e: (int(e.get("periodId") or 0), elapsed_seconds(e), int(e.get("eventId") or 0)))
        events_by_match[path.stem] = events
    return events_by_match


def event_lookup(events_by_match: dict[str, list[dict]]) -> dict[tuple[str, int], int]:
    lookup = {}
    for match_id, events in events_by_match.items():
        for idx, event in enumerate(events):
            lookup[(match_id, int(event.get("eventId") or -1))] = idx
    return lookup


def first_touch_after(events: list[dict], start_idx: int, clearance_team: str, max_seconds: float = 10.0) -> dict:
    clearance = events[start_idx]
    start_elapsed = elapsed_seconds(clearance)
    period = int(clearance.get("periodId") or 0)
    first_touch = None
    opponent_shot = False
    opponent_final_third_touch = False
    same_team_touch = False
    opponent_touch = False

    for nxt in events[start_idx + 1 :]:
        if int(nxt.get("periodId") or 0) != period:
            break
        dt = elapsed_seconds(nxt) - start_elapsed
        if dt < 0:
            continue
        if dt > max_seconds:
            break

        nxt_type = int(nxt.get("typeId") or -1)
        nxt_team = str(nxt.get("contestantId") or "")
        if nxt_team and nxt_team != clearance_team and nxt_type in SHOT_TYPES:
            opponent_shot = True

        if nxt_type in TOUCH_EXCLUDE_TYPES or not nxt_team:
            continue

        if nxt_team == clearance_team:
            same_team_touch = True
        else:
            opponent_touch = True
            if float(nxt.get("x") or 0) >= 66.667:
                opponent_final_third_touch = True

        if first_touch is None:
            first_touch = nxt

    if first_touch is None:
        return {
            "next_touch_team": "none",
            "next_touch_type": np.nan,
            "next_touch_player": "",
            "next_touch_seconds": np.nan,
            "same_team_first_touch": False,
            "opponent_first_touch": False,
            "same_team_touch_10s": same_team_touch,
            "opponent_touch_10s": opponent_touch,
            "opponent_shot_10s": opponent_shot,
            "opponent_final_third_touch_10s": opponent_final_third_touch,
        }

    first_team = str(first_touch.get("contestantId") or "")
    return {
        "next_touch_team": "same" if first_team == clearance_team else "opponent",
        "next_touch_type": int(first_touch.get("typeId") or -1),
        "next_touch_player": first_touch.get("playerName") or "",
        "next_touch_seconds": elapsed_seconds(first_touch) - start_elapsed,
        "same_team_first_touch": first_team == clearance_team,
        "opponent_first_touch": first_team != clearance_team,
        "same_team_touch_10s": same_team_touch,
        "opponent_touch_10s": opponent_touch,
        "opponent_shot_10s": opponent_shot,
        "opponent_final_third_touch_10s": opponent_final_third_touch,
    }


def add_predictions_and_insights(df: pd.DataFrame, events_by_match: dict[str, list[dict]]) -> pd.DataFrame:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    pred = model.predict(df[feature_cols])

    out = df.copy()
    out["pred_landing_x"] = np.clip(pred[:, 0], 0, 100)
    out["pred_landing_y"] = np.clip(pred[:, 1], 0, 100)
    out["expected_length"] = np.sqrt(
        (out["pred_landing_x"] - out["start_x"]) ** 2
        + (out["pred_landing_y"] - out["start_y"]) ** 2
    )
    out["landing_error"] = np.sqrt(
        (out["landing_x"] - out["pred_landing_x"]) ** 2
        + (out["landing_y"] - out["pred_landing_y"]) ** 2
    )
    out["length_oe"] = out["clearance_length"] - out["expected_length"]
    out["territory_oe"] = out["landing_x"] - out["pred_landing_x"]
    out["wide_oe"] = (out["landing_y"] - 50).abs() - (out["pred_landing_y"] - 50).abs()
    out["actual_zone"] = [landing_zone(x, y) for x, y in zip(out["landing_x"], out["landing_y"])]
    out["expected_zone"] = [landing_zone(x, y) for x, y in zip(out["pred_landing_x"], out["pred_landing_y"])]
    out["actual_danger_zone"] = [is_danger_zone(x, y) for x, y in zip(out["landing_x"], out["landing_y"])]
    out["expected_danger_zone"] = [is_danger_zone(x, y) for x, y in zip(out["pred_landing_x"], out["pred_landing_y"])]
    out["actual_safe_wide_zone"] = [is_safe_wide_zone(x, y) for x, y in zip(out["landing_x"], out["landing_y"])]
    out["expected_safe_wide_zone"] = [is_safe_wide_zone(x, y) for x, y in zip(out["pred_landing_x"], out["pred_landing_y"])]
    out["danger_zone_avoidance"] = out["expected_danger_zone"].astype(int) - out["actual_danger_zone"].astype(int)
    out["safe_wide_oe"] = out["actual_safe_wide_zone"].astype(int) - out["expected_safe_wide_zone"].astype(int)

    lookup = event_lookup(events_by_match)
    next_rows = []
    for _, row in out.iterrows():
        events = events_by_match.get(row["match_id"], [])
        idx = lookup.get((row["match_id"], int(row["event_id"])))
        if idx is None:
            next_rows.append({})
            continue
        next_rows.append(first_touch_after(events, idx, str(row["contestant_id"])))
    next_df = pd.DataFrame(next_rows)
    out = pd.concat([out.reset_index(drop=True), next_df.reset_index(drop=True)], axis=1)

    out["relief_success"] = (
        out["same_team_first_touch"].fillna(False)
        | out["actual_safe_wide_zone"].fillna(False)
        | (out["landing_x"] >= 45)
    ) & ~out["opponent_shot_10s"].fillna(False)

    out["clearance_value_oe"] = (
        0.35 * out["territory_oe"]
        + 0.25 * out["length_oe"]
        + 0.12 * out["wide_oe"]
        + 7.5 * out["same_team_first_touch"].astype(float)
        + 5.0 * out["danger_zone_avoidance"].astype(float)
        + 3.5 * out["safe_wide_oe"].astype(float)
        - 9.0 * out["opponent_first_touch"].astype(float)
        - 14.0 * out["opponent_shot_10s"].astype(float)
        - 4.0 * out["opponent_final_third_touch_10s"].astype(float)
    )
    return out


def summarize(grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    summary = grouped.agg(
        headed_clearances=("event_id", "count"),
        mean_landing_error=("landing_error", "mean"),
        median_landing_error=("landing_error", "median"),
        actual_length=("clearance_length", "mean"),
        expected_length=("expected_length", "mean"),
        length_oe=("length_oe", "mean"),
        territory_oe=("territory_oe", "mean"),
        wide_oe=("wide_oe", "mean"),
        danger_zone_rate=("actual_danger_zone", "mean"),
        expected_danger_zone_rate=("expected_danger_zone", "mean"),
        danger_zone_avoidance=("danger_zone_avoidance", "mean"),
        safe_wide_rate=("actual_safe_wide_zone", "mean"),
        expected_safe_wide_rate=("expected_safe_wide_zone", "mean"),
        same_team_first_touch_rate=("same_team_first_touch", "mean"),
        opponent_first_touch_rate=("opponent_first_touch", "mean"),
        opponent_shot_10s_rate=("opponent_shot_10s", "mean"),
        opponent_final_third_touch_10s_rate=("opponent_final_third_touch_10s", "mean"),
        relief_success_rate=("relief_success", "mean"),
        clearance_value_oe=("clearance_value_oe", "mean"),
    ).reset_index()
    return summary.sort_values(["clearance_value_oe", "headed_clearances"], ascending=[False, False])


def save_value_pitch(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)
    vmax = max(abs(player["clearance_value_oe"].min()), abs(player["clearance_value_oe"].max()), 1)
    scatter = pitch.scatter(
        player["landing_x"],
        player["landing_y"],
        ax=ax,
        s=np.clip(player["clearance_length"] * 5.0, 60, 250),
        c=player["clearance_value_oe"],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolors="white",
        lw=0.55,
        alpha=0.92,
        zorder=4,
    )
    for _, row in player.iterrows():
        color = "#7ee081" if row["clearance_value_oe"] >= 0 else "#ff6b6b"
        pitch.lines(row["start_x"], row["start_y"], row["landing_x"], row["landing_y"], ax=ax, color=color, alpha=0.25, lw=1.0)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Clearance value over expected", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    ax.set_title(f"{player.iloc[0]['player_name']} | Clearance Value Over Expected", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_outcome_pitch(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)
    colors = np.where(player["same_team_first_touch"], "#7ee081", np.where(player["opponent_shot_10s"], "#ff4d6d", "#ffbf69"))
    pitch.scatter(player["landing_x"], player["landing_y"], ax=ax, s=105, c=colors, edgecolors="white", lw=0.7, alpha=0.9, zorder=4)
    pitch.scatter(player["start_x"], player["start_y"], ax=ax, s=45, c="#9fb2c3", edgecolors="none", alpha=0.65, zorder=3)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7ee081", markeredgecolor="white", markersize=9, label="Same team first touch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffbf69", markeredgecolor="white", markersize=9, label="Opponent first touch"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff4d6d", markeredgecolor="white", markersize=9, label="Opponent shot within 10s"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, labelcolor="#f8fafc")
    ax.set_title(f"{player.iloc[0]['player_name']} | What Happened After the Header?", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_over_expectation_bars(player_summary: pd.Series, league: pd.DataFrame, out_path: pathlib.Path) -> None:
    metrics = [
        ("territory_oe", "Territory OE"),
        ("length_oe", "Length OE"),
        ("wide_oe", "Wide OE"),
        ("danger_zone_avoidance", "Danger Avoid."),
        ("same_team_first_touch_rate", "Same Team 1st"),
        ("opponent_shot_10s_rate", "Opp. Shot 10s"),
        ("clearance_value_oe", "Value OE"),
    ]
    values = [float(player_summary[key]) for key, _ in metrics]
    league_values = [float(league[key].mean()) for key, _ in metrics]
    labels = [label for _, label in metrics]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    ax.barh(y + 0.18, league_values, height=0.32, color="#405160", label="League player average")
    ax.barh(y - 0.18, values, height=0.32, color="#ffbf69", label=str(player_summary["player_name"]))
    ax.axvline(0, color="#d7dde2", lw=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fafc")
    ax.tick_params(axis="x", colors="#d7dde2")
    ax.grid(axis="x", color="#405160", alpha=0.35, lw=0.7)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.legend(frameon=False, labelcolor="#f8fafc", loc="lower right")
    ax.set_title(f"{player_summary['player_name']} | Headed Clearance Insight Profile", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_zone_matrix(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    matrix = pd.crosstab(player["expected_zone"], player["actual_zone"])
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    im = ax.imshow(matrix.to_numpy(), cmap="YlOrBr")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", color="#f8fafc")
    ax.set_yticklabels(matrix.index, color="#f8fafc")
    ax.tick_params(colors="#f8fafc")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", color="#101820", fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    ax.set_title(f"{player.iloc[0]['player_name']} | Expected Zone vs Actual Zone", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_rankings(player_summary: pd.DataFrame, out_path: pathlib.Path, min_clearances: int = 10) -> None:
    ranked = player_summary[player_summary["headed_clearances"] >= min_clearances].copy()
    ranked = ranked.sort_values("clearance_value_oe", ascending=False)
    top = ranked.head(12)
    bottom = ranked.tail(12).sort_values("clearance_value_oe")
    plot_df = pd.concat([bottom, top], ignore_index=True)
    colors = np.where(plot_df["clearance_value_oe"] >= 0, "#7ee081", "#ff6b6b")

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    labels = [f"{r.player_name} ({int(r.headed_clearances)})" for r in plot_df.itertuples()]
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["clearance_value_oe"], color=colors, alpha=0.85)
    ax.axvline(0, color="#d7dde2", lw=1.0, alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fafc", fontsize=9)
    ax.tick_params(axis="x", colors="#d7dde2")
    ax.grid(axis="x", color="#405160", alpha=0.35, lw=0.7)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.set_title(
        f"Headed Clearance Value Over Expected | Top and Bottom Players, min {min_clearances}",
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Clearance value over expected per headed clearance", color="#d7dde2")
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_team_rankings(team_summary: pd.DataFrame, out_path: pathlib.Path) -> None:
    ranked = team_summary.sort_values("clearance_value_oe", ascending=True).copy()
    colors = np.where(ranked["clearance_value_oe"] >= 0, "#7ee081", "#ff6b6b")

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    labels = [f"{r.team} ({int(r.headed_clearances)})" for r in ranked.itertuples()]
    y = np.arange(len(ranked))
    ax.barh(y, ranked["clearance_value_oe"], color=colors, alpha=0.85)
    ax.axvline(0, color="#d7dde2", lw=1.0, alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fafc", fontsize=9)
    ax.tick_params(axis="x", colors="#d7dde2")
    ax.grid(axis="x", color="#405160", alpha=0.35, lw=0.7)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.set_title("Team Headed Clearance Value Over Expected", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Clearance value over expected per headed clearance", color="#d7dde2")
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_readme(out_path: pathlib.Path) -> None:
    out_path.write_text(
        """# Headed Clearance Landing Insights

This folder turns the headed-clearance landing model into analysis outputs.

## Core files

- `headed_clearance_insights.csv`: one row per headed clearance with model prediction, actual landing, over-expectation metrics, and next-action outcome flags.
- `headed_clearance_player_summary.csv`: player-level summary.
- `headed_clearance_team_summary.csv`: team-level summary.
- `headed_clearance_zone_summary.csv`: summary by actual landing zone.

## Main metrics

- `length_oe`: actual clearance length minus expected clearance length.
- `territory_oe`: actual landing x minus predicted landing x. Positive means further upfield than expected.
- `wide_oe`: actual landing distance from the pitch centerline minus expected distance from centerline. Positive means wider than expected.
- `danger_zone_avoidance`: 1 when the model expected a defensive-central landing but the actual clearance avoided it.
- `same_team_first_touch_rate`: how often the clearing team got the first recorded touch within 10 seconds.
- `opponent_shot_10s_rate`: how often the opponent shot within 10 seconds.
- `clearance_value_oe`: weighted composite combining territory, length, width, first-touch outcome, danger-zone avoidance, and danger after the clearance.

## Interpretation

Use `clearance_value_oe` as a directional analysis score, not as a universal truth metric. It is built for comparing headed clearances in this Ecuador 2026 event feed and should be reviewed alongside the maps and next-action context.
""",
        encoding="utf-8",
    )


def save_player_visuals(insights: pd.DataFrame, player_summary: pd.DataFrame, out_dir: pathlib.Path, player_name: str) -> list[pathlib.Path]:
    player = insights[insights["player_name"].str.casefold() == player_name.casefold()].copy()
    if player.empty:
        raise ValueError(f"No headed clearances found for {player_name}")
    row = player_summary[player_summary["player_name"].str.casefold() == player_name.casefold()].iloc[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(player.iloc[0]["player_name"])
    paths = [
        out_dir / f"{prefix}_value_over_expected_map.png",
        out_dir / f"{prefix}_next_outcome_map.png",
        out_dir / f"{prefix}_insight_profile.png",
        out_dir / f"{prefix}_expected_vs_actual_zone_matrix.png",
    ]
    save_value_pitch(player, paths[0])
    save_outcome_pitch(player, paths[1])
    save_over_expectation_bars(row, player_summary[player_summary["headed_clearances"] >= 5], paths[2])
    save_zone_matrix(player, paths[3])
    return paths


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out_dir / "player_visuals"

    base = pd.read_csv(DATASET_PATH)
    events_by_match = load_match_events(args.data_dir)
    insights = add_predictions_and_insights(base, events_by_match)

    event_path = args.out_dir / "headed_clearance_insights.csv"
    player_summary_path = args.out_dir / "headed_clearance_player_summary.csv"
    team_summary_path = args.out_dir / "headed_clearance_team_summary.csv"
    zone_summary_path = args.out_dir / "headed_clearance_zone_summary.csv"
    meta_path = args.out_dir / "headed_clearance_insights_meta.json"

    insights.to_csv(event_path, index=False)
    player_summary = summarize(insights.groupby(["player_id", "player_name", "team"]))
    team_summary = summarize(insights.groupby(["team"]))
    zone_summary = summarize(insights.groupby(["actual_zone"]))
    player_summary.to_csv(player_summary_path, index=False)
    team_summary.to_csv(team_summary_path, index=False)
    zone_summary.to_csv(zone_summary_path, index=False)

    visual_paths = save_player_visuals(insights, player_summary, vis_dir, args.player)
    league_visual_paths = [
        args.out_dir / "player_value_rankings.png",
        args.out_dir / "team_value_rankings.png",
    ]
    save_player_rankings(player_summary, league_visual_paths[0])
    save_team_rankings(team_summary, league_visual_paths[1])
    readme_path = args.out_dir / "README.md"
    save_readme(readme_path)

    meta = {
        "rows": int(len(insights)),
        "matches": int(insights["match_id"].nunique()),
        "player_visualized": args.player,
        "outputs": {
            "event_insights": str(event_path),
            "player_summary": str(player_summary_path),
            "team_summary": str(team_summary_path),
            "zone_summary": str(zone_summary_path),
            "visuals": [str(p) for p in [*visual_paths, *league_visual_paths]],
            "readme": str(readme_path),
        },
        "metric_notes": {
            "length_oe": "actual clearance length minus model-expected clearance length",
            "territory_oe": "actual landing x minus predicted landing x",
            "wide_oe": "actual distance from pitch centerline minus predicted distance from centerline",
            "clearance_value_oe": "weighted composite of territory, length, width, first touch, danger avoidance, and danger after clearance",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for path in [event_path, player_summary_path, team_summary_path, zone_summary_path, meta_path, readme_path, *visual_paths, *league_visual_paths]:
        print(path)


if __name__ == "__main__":
    main()
