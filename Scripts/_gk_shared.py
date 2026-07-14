"""
Shared constants and data-prep helpers for the goalkeeper-model advanced
visuals / regression / Monte Carlo scripts. Not a standalone script --
imported by create_goalkeeper_advanced_visuals.py and
create_goalkeeper_regression_montecarlo.py.
"""

from __future__ import annotations

import glob
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "Lamberts Goalkeeper Model"
DANGER_DIR = ROOT / "Danger"
VIS_DIR = MODEL_DIR / "visuals"

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

BG = "#0d1117"
PANEL_BG = "#101820"
PITCH_BG = "#101820"
GRID_COLOR = "#405160"
TEXT_MAIN = "#f8fafc"
TEXT_SUB = "#9aa4b2"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_AMBER = "#ffc247"
GREEN = "#7ee081"
RED = "#ff6b6b"

MIN_MINUTES_FOR_RANKING = 450.0


def add_logo(fig, width=0.12, margin=0.016):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")


def style_axes(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_SUB)
    ax.grid(color=GRID_COLOR, alpha=0.3, lw=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def load_season() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "goalkeeper_season_value_model.csv")


def load_match() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "goalkeeper_match_value.csv")


def load_raw_shots() -> pd.DataFrame:
    """All shots from Danger/*_danger_models.csv, tagged with match_file."""
    files = [f for f in glob.glob(str(DANGER_DIR / "*_danger_models.csv")) if "eredivisie" not in f]
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["match_file"] = pathlib.Path(f).name.replace("_danger_models.csv", ".json")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_shots_faced_by_keeper() -> pd.DataFrame:
    """Every individual shot, tagged with which keeper faced it (one row
    per shot-keeper pair; every match has exactly one keeper per team for
    the full 90, so this is an exact join, not an approximation)."""
    match_df = load_match()
    keeper_lookup = match_df[["match_file", "team_id", "player_id", "player", "team", "date", "minutes"]].rename(
        columns={
            "team_id": "defending_team_id", "player_id": "keeper_player_id",
            "player": "keeper", "team": "keeper_team", "minutes": "keeper_minutes",
        }
    )
    shots = load_raw_shots()
    merged = shots.merge(keeper_lookup, on="match_file", how="inner")
    faced = merged[merged["defending_team_id"] != merged["contestant_id"]].copy()
    return faced.reset_index(drop=True)
