"""
Create player visuals for headed-clearance landing predictions.

Example:
  python3 visualize_header_clearance_player.py "C. Gruezo"
"""

from __future__ import annotations

import argparse
import pathlib
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
OUT_DIR = MODEL_DIR / "player_visuals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create headed-clearance landing visuals for a player.")
    parser.add_argument("player_name")
    parser.add_argument("--out", type=pathlib.Path)
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


def save_overview(player: pd.DataFrame, bundle: dict, out_path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(15, 9), facecolor="#0b1117")
    ax = fig.add_axes([0.04, 0.08, 0.64, 0.82])
    pitch = make_pitch()
    pitch.draw(ax=ax)

    for _, row in player.iterrows():
        pitch.lines(
            row["start_x"],
            row["start_y"],
            row["landing_x"],
            row["landing_y"],
            ax=ax,
            color="#7aa7ff",
            alpha=0.22,
            lw=1.2,
            zorder=1,
        )
        pitch.lines(
            row["start_x"],
            row["start_y"],
            row["pred_landing_x"],
            row["pred_landing_y"],
            ax=ax,
            color="#ffbf69",
            alpha=0.20,
            lw=1.0,
            linestyle="--",
            zorder=1,
        )

    pitch.scatter(player["start_x"], player["start_y"], ax=ax, s=70, c="#f25f5c", edgecolors="white", lw=0.7, label="Header origin", zorder=4)
    pitch.scatter(player["landing_x"], player["landing_y"], ax=ax, s=64, c="#7aa7ff", edgecolors="white", lw=0.6, label="Actual landing", zorder=3)
    pitch.scatter(player["pred_landing_x"], player["pred_landing_y"], ax=ax, s=70, c="#ffbf69", marker="X", edgecolors="#111111", lw=0.7, label="Predicted landing", zorder=5)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.065), ncol=3, frameon=False, labelcolor="#f2f5f7", fontsize=11)

    side = fig.add_axes([0.72, 0.12, 0.24, 0.76])
    side.set_facecolor("#0b1117")
    side.axis("off")

    player_name = player.iloc[0]["player_name"]
    team = player["team"].mode().iloc[0] if len(player["team"].mode()) else "Unknown"
    n = len(player)
    median_origin = (player["start_x"].median(), player["start_y"].median())
    median_actual = (player["landing_x"].median(), player["landing_y"].median())
    median_pred = (player["pred_landing_x"].median(), player["pred_landing_y"].median())
    mean_error = player["landing_error"].mean()
    model_error = bundle["metrics"]["test"]["mean_landing_error"]

    side.text(0, 0.96, player_name, color="#f8fafc", fontsize=26, fontweight="bold", va="top")
    side.text(0, 0.90, team, color="#9fb2c3", fontsize=13, va="top")
    side.text(0, 0.78, "Headed Clearance Landing Model", color="#f8fafc", fontsize=14, fontweight="bold")

    stats = [
        ("Headed clearances", f"{n}"),
        ("Median origin", f"x={median_origin[0]:.1f}, y={median_origin[1]:.1f}"),
        ("Median actual landing", f"x={median_actual[0]:.1f}, y={median_actual[1]:.1f}"),
        ("Median predicted landing", f"x={median_pred[0]:.1f}, y={median_pred[1]:.1f}"),
        ("Player mean error", f"{mean_error:.1f} pitch units"),
        ("Model test mean error", f"{model_error:.1f} pitch units"),
    ]

    y = 0.68
    for label, value in stats:
        side.text(0, y, label.upper(), color="#7d90a1", fontsize=9, fontweight="bold")
        side.text(0, y - 0.04, value, color="#f8fafc", fontsize=16)
        y -= 0.105

    side.text(
        0,
        0.04,
        "Blue lines show actual headed-clearance flights. Dashed amber lines show the model prediction from the same origin.",
        color="#9fb2c3",
        fontsize=10,
        wrap=True,
    )

    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_landing_heatmaps(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    pitch = make_pitch()
    fig, axes = pitch.grid(
        ncols=2,
        figheight=7.8,
        title_height=0.08,
        endnote_height=0.03,
        axis=False,
    )
    fig.set_facecolor("#0b1117")
    axes["title"].text(
        0.5,
        0.5,
        f"{player.iloc[0]['player_name']} | Headed Clearance Landing Heatmaps",
        ha="center",
        va="center",
        color="#f8fafc",
        fontsize=18,
        fontweight="bold",
    )

    panels = [
        ("Actual landings", "landing_x", "landing_y", "Blues"),
        ("Predicted landings", "pred_landing_x", "pred_landing_y", "YlOrBr"),
    ]
    for ax, (title, x_col, y_col, cmap) in zip(axes["pitch"].flat, panels):
        pitch.draw(ax=ax)
        bin_stat = pitch.bin_statistic(player[x_col], player[y_col], statistic="count", bins=(12, 8))
        pitch.heatmap(bin_stat, ax=ax, cmap=cmap, edgecolors="#101820", alpha=0.86)
        pitch.scatter(player[x_col], player[y_col], ax=ax, s=26, c="#f8fafc", alpha=0.55, edgecolors="none")
        ax.set_title(title, color="#f8fafc", fontsize=13, fontweight="bold", pad=8)

    axes["endnote"].text(
        0.5,
        0.5,
        "Pitch type: mplsoccer Opta. Darker cells show more headed-clearance landings.",
        ha="center",
        va="center",
        color="#9fb2c3",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_error_map(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)

    sizes = np.interp(player["landing_error"], (player["landing_error"].min(), player["landing_error"].max()), (70, 260))
    scatter = pitch.scatter(
        player["landing_x"],
        player["landing_y"],
        ax=ax,
        s=sizes,
        c=player["landing_error"],
        cmap="magma_r",
        edgecolors="white",
        lw=0.5,
        alpha=0.88,
        zorder=4,
    )
    for _, row in player.iterrows():
        pitch.lines(
            row["landing_x"],
            row["landing_y"],
            row["pred_landing_x"],
            row["pred_landing_y"],
            ax=ax,
            color="#f8fafc",
            alpha=0.22,
            lw=0.9,
            zorder=2,
        )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Prediction error, pitch units", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    ax.set_title(
        f"{player.iloc[0]['player_name']} | Prediction Error by Actual Landing",
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_sequence_bars(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    chart = player.sort_values(["match_id", "period_id", "elapsed_seconds"]).copy()
    chart["event_no"] = np.arange(1, len(chart) + 1)

    fig, ax = plt.subplots(figsize=(12, 5.8), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    ax.bar(chart["event_no"], chart["clearance_length"], color="#7aa7ff", alpha=0.58, label="Actual length")
    ax.plot(chart["event_no"], chart["landing_error"], color="#ffbf69", lw=2.0, marker="o", ms=4, label="Prediction error")
    ax.axhline(chart["landing_error"].mean(), color="#f25f5c", lw=1.6, ls="--", label="Mean error")
    ax.set_title(
        f"{chart.iloc[0]['player_name']} | Clearance Length and Model Error",
        color="#f8fafc",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Headed clearance number", color="#d7dde2")
    ax.set_ylabel("Pitch units", color="#d7dde2")
    ax.tick_params(colors="#d7dde2")
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.grid(color="#405160", alpha=0.32, lw=0.7)
    ax.legend(frameon=False, labelcolor="#f8fafc", loc="upper left", ncol=3)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_origin_heatmap(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    pitch = make_pitch()
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch.draw(ax=ax)
    bin_stat = pitch.bin_statistic(player["start_x"], player["start_y"], statistic="count", bins=(12, 8))
    pitch.heatmap(bin_stat, ax=ax, cmap="Reds", edgecolors="#101820", alpha=0.86)
    pitch.scatter(player["start_x"], player["start_y"], ax=ax, s=48, c="#f8fafc", alpha=0.72, edgecolors="#101820", lw=0.4)
    ax.set_title(
        f"{player.iloc[0]['player_name']} | Headed Clearance Origins",
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_clearance_length_map(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)
    scatter = pitch.scatter(
        player["start_x"],
        player["start_y"],
        ax=ax,
        s=np.clip(player["clearance_length"] * 5.5, 60, 240),
        c=player["clearance_length"],
        cmap="viridis",
        edgecolors="white",
        lw=0.55,
        alpha=0.9,
        zorder=4,
    )
    for _, row in player.iterrows():
        pitch.lines(
            row["start_x"],
            row["start_y"],
            row["landing_x"],
            row["landing_y"],
            ax=ax,
            color="#d7dde2",
            alpha=0.20,
            lw=1.0,
            zorder=2,
        )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Clearance length, pitch units", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    ax.set_title(
        f"{player.iloc[0]['player_name']} | Clearance Length from Origin",
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_coordinate_fit(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), facecolor="#0b1117", constrained_layout=True)
    panels = [
        ("Landing X", "landing_x", "pred_landing_x", "#7aa7ff"),
        ("Landing Y", "landing_y", "pred_landing_y", "#ffbf69"),
    ]
    for ax, (title, actual_col, pred_col, color) in zip(axes, panels):
        ax.set_facecolor("#101820")
        ax.scatter(player[actual_col], player[pred_col], s=78, color=color, edgecolors="white", lw=0.5, alpha=0.86)
        ax.plot([0, 100], [0, 100], color="#f25f5c", lw=1.5, ls="--")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold")
        ax.set_xlabel("Actual", color="#d7dde2")
        ax.set_ylabel("Predicted", color="#d7dde2")
        ax.tick_params(colors="#d7dde2")
        ax.grid(color="#405160", alpha=0.35, lw=0.7)
        for spine in ax.spines.values():
            spine.set_color("#405160")
    fig.suptitle(
        f"{player.iloc[0]['player_name']} | Actual vs Predicted Landing Coordinates",
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_vs_league_distributions(player: pd.DataFrame, all_events: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor="#0b1117", constrained_layout=True)
    panels = [
        ("Clearance Length", "clearance_length", "Pitch units"),
        ("Prediction Error", "landing_error", "Pitch units"),
    ]
    for ax, (title, col, xlabel) in zip(axes, panels):
        ax.set_facecolor("#101820")
        ax.hist(all_events[col].dropna(), bins=24, color="#405160", alpha=0.82, density=True, label="League headed clearances")
        ax.hist(player[col].dropna(), bins=12, color="#ffbf69", alpha=0.78, density=True, label=player.iloc[0]["player_name"])
        ax.axvline(player[col].median(), color="#f25f5c", lw=2.0, label="Player median")
        ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, color="#d7dde2")
        ax.set_ylabel("Density", color="#d7dde2")
        ax.tick_params(colors="#d7dde2")
        ax.grid(color="#405160", alpha=0.35, lw=0.7)
        for spine in ax.spines.values():
            spine.set_color("#405160")
    axes[0].legend(frameon=False, labelcolor="#f8fafc", loc="upper right")
    fig.suptitle(
        f"{player.iloc[0]['player_name']} | Compared with Ecuador 2026 Headed Clearances",
        color="#f8fafc",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_csv(DATASET_PATH)
    player = df[df["player_name"].str.casefold() == args.player_name.casefold()].copy()
    if player.empty:
        close = df[df["player_name"].str.contains(args.player_name, case=False, na=False)]["player_name"].drop_duplicates()
        if len(close):
            print("No exact player match. Possible matches:")
            for name in close.head(20):
                print(f"  - {name}")
        else:
            print(f"No headed-clearance records found for '{args.player_name}'.")
        raise SystemExit(1)

    pred = model.predict(player[feature_cols])
    player["pred_landing_x"] = np.clip(pred[:, 0], 0, 100)
    player["pred_landing_y"] = np.clip(pred[:, 1], 0, 100)
    player["landing_error"] = np.sqrt(
        (player["landing_x"] - player["pred_landing_x"]) ** 2
        + (player["landing_y"] - player["pred_landing_y"]) ** 2
    )

    all_pred = model.predict(df[feature_cols])
    df["pred_landing_x"] = np.clip(all_pred[:, 0], 0, 100)
    df["pred_landing_y"] = np.clip(all_pred[:, 1], 0, 100)
    df["landing_error"] = np.sqrt(
        (df["landing_x"] - df["pred_landing_x"]) ** 2
        + (df["landing_y"] - df["pred_landing_y"]) ** 2
    )

    prefix = safe_name(player.iloc[0]["player_name"])
    paths = [
        args.out or OUT_DIR / f"{prefix}_headed_clearance_landing.png",
        OUT_DIR / f"{prefix}_landing_heatmaps.png",
        OUT_DIR / f"{prefix}_prediction_error_map.png",
        OUT_DIR / f"{prefix}_clearance_error_timeline.png",
        OUT_DIR / f"{prefix}_origin_heatmap.png",
        OUT_DIR / f"{prefix}_clearance_length_map.png",
        OUT_DIR / f"{prefix}_coordinate_fit.png",
        OUT_DIR / f"{prefix}_player_vs_league_distributions.png",
    ]

    save_overview(player, bundle, paths[0])
    save_landing_heatmaps(player, paths[1])
    save_error_map(player, paths[2])
    save_sequence_bars(player, paths[3])
    save_origin_heatmap(player, paths[4])
    save_clearance_length_map(player, paths[5])
    save_coordinate_fit(player, paths[6])
    save_player_vs_league_distributions(player, df, paths[7])

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
