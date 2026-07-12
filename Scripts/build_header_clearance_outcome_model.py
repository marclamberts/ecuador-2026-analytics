"""
Train outcome models for headed clearances and create decision-quality visuals.

This sits on top of the landing model and insight layer:
  - predicts P(relief success)
  - predicts P(opponent shot within 10 seconds)
  - adds pre-header context features
  - clusters headed-clearance styles
  - creates player and league visuals

Example:
  python3 build_header_clearance_outcome_model.py --player "C. Gruezo"
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = pathlib.Path(__file__).resolve().parent
INSIGHTS_DIR = HERE / "ClearanceLandingModel" / "Insights"
INSIGHTS_PATH = INSIGHTS_DIR / "headed_clearance_insights.csv"
OUT_DIR = HERE / "ClearanceLandingModel" / "OutcomeModel"
VIS_DIR = OUT_DIR / "visuals"

ADMIN_TYPES = {18, 19, 30, 32, 34, 37, 40, 70, 71, 90, 91}
TOUCH_EXCLUDE_TYPES = ADMIN_TYPES | {27, 28}
SHOT_TYPES = {13, 14, 15, 16}
SET_PIECE_QIDS = {5, 6, 107}
CROSS_QID = 2
THROUGH_BALL_QID = 4
LONG_BALL_QID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train headed-clearance outcome models.")
    parser.add_argument("--player", default="C. Gruezo")
    parser.add_argument("--data-dir", type=pathlib.Path, default=HERE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
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


def qids(event: dict) -> set[int]:
    return {int(q["qualifierId"]) for q in event.get("qualifier", [])}


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


def pre_header_context(events: list[dict], idx: int, team_id: str, max_seconds: float = 15.0) -> dict:
    clearance = events[idx]
    start_elapsed = elapsed_seconds(clearance)
    period = int(clearance.get("periodId") or 0)

    prev_touch = None
    recent_opponent_touches = 0
    recent_same_touches = 0
    recent_shots = 0
    recent_crosses = 0
    recent_set_pieces = 0
    recent_final_third_touches = 0
    recent_pressure_events = 0

    for prev in reversed(events[:idx]):
        if int(prev.get("periodId") or 0) != period:
            break
        dt = start_elapsed - elapsed_seconds(prev)
        if dt < 0:
            continue
        if dt > max_seconds:
            break

        prev_type = int(prev.get("typeId") or -1)
        prev_team = str(prev.get("contestantId") or "")
        prev_qids = qids(prev)

        if prev_type == 61:
            recent_pressure_events += 1
        if prev_type in SHOT_TYPES:
            recent_shots += 1
        if CROSS_QID in prev_qids:
            recent_crosses += 1
        if SET_PIECE_QIDS & prev_qids:
            recent_set_pieces += 1
        if prev_team and prev_team != team_id:
            recent_opponent_touches += 1
            if float(prev.get("x") or 0) >= 66.667:
                recent_final_third_touches += 1
        elif prev_team == team_id:
            recent_same_touches += 1

        if prev_touch is None and prev_type not in TOUCH_EXCLUDE_TYPES and prev_team:
            prev_touch = prev

    if prev_touch is None:
        return {
            "prev_type": "none",
            "prev_team_relation": "none",
            "prev_outcome": -1,
            "prev_x": 50.0,
            "prev_y": 50.0,
            "prev_seconds": max_seconds,
            "prev_was_cross": False,
            "prev_was_through_ball": False,
            "prev_was_long_ball": False,
            "prev_was_set_piece": False,
            "prev_was_shot": False,
            "recent_opponent_touches_15s": recent_opponent_touches,
            "recent_same_touches_15s": recent_same_touches,
            "recent_shots_15s": recent_shots,
            "recent_crosses_15s": recent_crosses,
            "recent_set_pieces_15s": recent_set_pieces,
            "recent_final_third_touches_15s": recent_final_third_touches,
            "recent_pressure_events_15s": recent_pressure_events,
        }

    prev_team = str(prev_touch.get("contestantId") or "")
    prev_type = int(prev_touch.get("typeId") or -1)
    prev_qids = qids(prev_touch)
    return {
        "prev_type": str(prev_type),
        "prev_team_relation": "same" if prev_team == team_id else "opponent",
        "prev_outcome": int(prev_touch.get("outcome") if prev_touch.get("outcome") is not None else -1),
        "prev_x": float(prev_touch.get("x") if prev_touch.get("x") is not None else 50.0),
        "prev_y": float(prev_touch.get("y") if prev_touch.get("y") is not None else 50.0),
        "prev_seconds": start_elapsed - elapsed_seconds(prev_touch),
        "prev_was_cross": CROSS_QID in prev_qids,
        "prev_was_through_ball": THROUGH_BALL_QID in prev_qids,
        "prev_was_long_ball": LONG_BALL_QID in prev_qids,
        "prev_was_set_piece": bool(SET_PIECE_QIDS & prev_qids),
        "prev_was_shot": prev_type in SHOT_TYPES,
        "recent_opponent_touches_15s": recent_opponent_touches,
        "recent_same_touches_15s": recent_same_touches,
        "recent_shots_15s": recent_shots,
        "recent_crosses_15s": recent_crosses,
        "recent_set_pieces_15s": recent_set_pieces,
        "recent_final_third_touches_15s": recent_final_third_touches,
        "recent_pressure_events_15s": recent_pressure_events,
    }


def add_context(insights: pd.DataFrame, events_by_match: dict[str, list[dict]]) -> pd.DataFrame:
    lookup = event_lookup(events_by_match)
    rows = []
    for _, row in insights.iterrows():
        events = events_by_match.get(row["match_id"], [])
        idx = lookup.get((row["match_id"], int(row["event_id"])))
        if idx is None:
            rows.append({})
        else:
            rows.append(pre_header_context(events, idx, str(row["contestant_id"])))
    return pd.concat([insights.reset_index(drop=True), pd.DataFrame(rows).reset_index(drop=True)], axis=1)


def make_classifier(random_state: int) -> Pipeline:
    numeric = [
        "period_id",
        "elapsed_seconds",
        "start_x",
        "start_y",
        "distance_from_center",
        "pred_landing_x",
        "pred_landing_y",
        "expected_length",
        "prev_x",
        "prev_y",
        "prev_seconds",
        "recent_opponent_touches_15s",
        "recent_same_touches_15s",
        "recent_shots_15s",
        "recent_crosses_15s",
        "recent_set_pieces_15s",
        "recent_final_third_touches_15s",
        "recent_pressure_events_15s",
    ]
    categorical = [
        "team",
        "player_id",
        "start_x_zone",
        "start_y_zone",
        "direction",
        "expected_zone",
        "prev_type",
        "prev_team_relation",
        "prev_outcome",
        "prev_was_cross",
        "prev_was_through_ball",
        "prev_was_long_ball",
        "prev_was_set_piece",
        "prev_was_shot",
    ]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse=False)
    pre = ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("categorical", encoder, categorical)],
        remainder="drop",
    )
    clf = RandomForestClassifier(
        n_estimators=450,
        min_samples_leaf=8,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", clf)])


def safe_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "brier": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
        "mean_predicted_probability": float(np.mean(proba)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        metrics["log_loss"] = float(log_loss(y_true, np.column_stack([1 - proba, proba]), labels=[0, 1]))
    else:
        metrics["roc_auc"] = np.nan
        metrics["log_loss"] = np.nan
    return metrics


def train_outcome_models(df: pd.DataFrame, random_state: int) -> tuple[dict, pd.DataFrame, dict]:
    target_cols = {
        "relief_success": "p_relief_success",
        "opponent_shot_10s": "p_opponent_shot_10s",
    }
    feature_cols = [
        "period_id",
        "elapsed_seconds",
        "start_x",
        "start_y",
        "distance_from_center",
        "pred_landing_x",
        "pred_landing_y",
        "expected_length",
        "team",
        "player_id",
        "start_x_zone",
        "start_y_zone",
        "direction",
        "expected_zone",
        "prev_x",
        "prev_y",
        "prev_seconds",
        "prev_type",
        "prev_team_relation",
        "prev_outcome",
        "prev_was_cross",
        "prev_was_through_ball",
        "prev_was_long_ball",
        "prev_was_set_piece",
        "prev_was_shot",
        "recent_opponent_touches_15s",
        "recent_same_touches_15s",
        "recent_shots_15s",
        "recent_crosses_15s",
        "recent_set_pieces_15s",
        "recent_final_third_touches_15s",
        "recent_pressure_events_15s",
    ]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    models = {}
    metrics = {"n_train": int(len(train)), "n_test": int(len(test))}
    predictions = df.copy()
    predictions["is_test"] = False
    predictions.loc[test.index, "is_test"] = True

    for target, proba_col in target_cols.items():
        model = make_classifier(random_state)
        y_train = train[target].astype(bool).astype(int)
        y_test = test[target].astype(bool).astype(int)
        model.fit(train[feature_cols], y_train)
        predictions[proba_col] = model.predict_proba(df[feature_cols])[:, 1]
        test_proba = model.predict_proba(test[feature_cols])[:, 1]
        train_proba = model.predict_proba(train[feature_cols])[:, 1]
        models[target] = model
        metrics[target] = {
            "train": safe_metrics(y_train, train_proba),
            "test": safe_metrics(y_test, test_proba),
        }

    predictions["relief_oe"] = predictions["relief_success"].astype(int) - predictions["p_relief_success"]
    predictions["shot_prevention_oe"] = predictions["p_opponent_shot_10s"] - predictions["opponent_shot_10s"].astype(int)
    predictions["decision_quality_oe"] = (
        predictions["clearance_value_oe"]
        + 10.0 * predictions["relief_oe"]
        + 15.0 * predictions["shot_prevention_oe"]
    )

    bundle = {"models": models, "feature_cols": feature_cols, "target_cols": target_cols, "metrics": metrics}
    return bundle, predictions, metrics


def add_style_clusters(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    cluster_features = [
        "start_x",
        "start_y",
        "landing_x",
        "landing_y",
        "clearance_length",
        "territory_oe",
        "wide_oe",
        "p_relief_success",
        "p_opponent_shot_10s",
    ]
    work = df.copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(work[cluster_features])
    n_clusters = min(5, max(2, len(work) // 250))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    work["style_cluster"] = kmeans.fit_predict(x)

    summary = work.groupby("style_cluster").agg(
        clearances=("event_id", "count"),
        start_x=("start_x", "mean"),
        start_y=("start_y", "mean"),
        landing_x=("landing_x", "mean"),
        landing_y=("landing_y", "mean"),
        clearance_length=("clearance_length", "mean"),
        territory_oe=("territory_oe", "mean"),
        wide_oe=("wide_oe", "mean"),
        p_relief_success=("p_relief_success", "mean"),
        p_opponent_shot_10s=("p_opponent_shot_10s", "mean"),
        decision_quality_oe=("decision_quality_oe", "mean"),
    ).reset_index()

    names = {}
    for row in summary.itertuples():
        if row.start_x < 15 and abs(row.start_y - 50) < 16:
            label = "Deep central emergency"
        elif row.clearance_length >= summary["clearance_length"].quantile(0.75):
            label = "Long relief header"
        elif abs(row.landing_y - 50) >= 25:
            label = "Wide safety header"
        elif row.p_opponent_shot_10s >= summary["p_opponent_shot_10s"].quantile(0.75):
            label = "High-risk second wave"
        else:
            label = "Central contestable header"
        names[row.style_cluster] = f"{label} {int(row.style_cluster)}"

    work["style_name"] = work["style_cluster"].map(names)
    summary["style_name"] = summary["style_cluster"].map(names)
    return work, summary, kmeans


def summarize_players(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["player_id", "player_name", "team"]).agg(
        headed_clearances=("event_id", "count"),
        p_relief_success=("p_relief_success", "mean"),
        relief_success_rate=("relief_success", "mean"),
        relief_oe=("relief_oe", "mean"),
        p_opponent_shot_10s=("p_opponent_shot_10s", "mean"),
        opponent_shot_10s_rate=("opponent_shot_10s", "mean"),
        shot_prevention_oe=("shot_prevention_oe", "mean"),
        clearance_value_oe=("clearance_value_oe", "mean"),
        decision_quality_oe=("decision_quality_oe", "mean"),
        length_oe=("length_oe", "mean"),
        territory_oe=("territory_oe", "mean"),
        wide_oe=("wide_oe", "mean"),
    ).reset_index().sort_values(["decision_quality_oe", "headed_clearances"], ascending=[False, False])


def calibration_table(y_true: pd.Series, proba: pd.Series, bins: int = 10) -> pd.DataFrame:
    table = pd.DataFrame({"y": y_true.astype(int), "p": proba})
    table["bin"] = pd.cut(table["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return table.groupby("bin", observed=False).agg(
        n=("y", "size"),
        actual_rate=("y", "mean"),
        predicted_rate=("p", "mean"),
    ).reset_index()


def save_calibration_plot(df: pd.DataFrame, target: str, proba_col: str, title: str, out_path: pathlib.Path) -> None:
    cal = calibration_table(df[target], df[proba_col])
    fig, ax = plt.subplots(figsize=(7.5, 7), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    ax.plot([0, 1], [0, 1], color="#d7dde2", lw=1.2, ls="--")
    sizes = np.clip(cal["n"].fillna(0) * 5, 25, 260)
    ax.scatter(cal["predicted_rate"], cal["actual_rate"], s=sizes, color="#ffbf69", edgecolors="white", lw=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", color="#d7dde2")
    ax.set_ylabel("Actual rate", color="#d7dde2")
    ax.tick_params(colors="#d7dde2")
    ax.grid(color="#405160", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.set_title(title, color="#f8fafc", fontsize=16, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_outcome_map(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)
    vmax = max(abs(player["decision_quality_oe"].min()), abs(player["decision_quality_oe"].max()), 1)
    scatter = pitch.scatter(
        player["landing_x"],
        player["landing_y"],
        ax=ax,
        s=np.clip(player["clearance_length"] * 5, 70, 260),
        c=player["decision_quality_oe"],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolors="white",
        lw=0.6,
        alpha=0.92,
        zorder=4,
    )
    for _, row in player.iterrows():
        color = "#7ee081" if row["decision_quality_oe"] >= 0 else "#ff6b6b"
        pitch.lines(row["start_x"], row["start_y"], row["landing_x"], row["landing_y"], ax=ax, color=color, alpha=0.24, lw=1.1, zorder=2)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Decision quality over expected", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    ax.set_title(f"{player.iloc[0]['player_name']} | Decision Quality Over Expected", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_shot_risk_map(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)
    scatter = pitch.scatter(
        player["start_x"],
        player["start_y"],
        ax=ax,
        s=90 + player["p_opponent_shot_10s"] * 950,
        c=player["p_opponent_shot_10s"],
        cmap="magma_r",
        edgecolors="white",
        lw=0.65,
        alpha=0.92,
        zorder=4,
    )
    for _, row in player.iterrows():
        pitch.lines(row["start_x"], row["start_y"], row["pred_landing_x"], row["pred_landing_y"], ax=ax, color="#ffbf69", alpha=0.22, lw=1.1, linestyle="--")
        if row["opponent_shot_10s"]:
            pitch.scatter(row["landing_x"], row["landing_y"], ax=ax, marker="X", s=180, c="#ff4d6d", edgecolors="white", lw=0.8, zorder=5)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Predicted opponent shot risk within 10s", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    ax.set_title(f"{player.iloc[0]['player_name']} | Pre-Header Shot-Risk Model", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_style_map(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0b1117")
    pitch = make_pitch()
    pitch.draw(ax=ax)
    styles = sorted(player["style_name"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    for i, style in enumerate(styles):
        part = player[player["style_name"] == style]
        color = cmap(i % 10)
        for _, row in part.iterrows():
            pitch.lines(row["start_x"], row["start_y"], row["landing_x"], row["landing_y"], ax=ax, color=color, alpha=0.28, lw=1.2)
        pitch.scatter(part["landing_x"], part["landing_y"], ax=ax, s=95, c=[color], edgecolors="white", lw=0.6, label=style, zorder=4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, labelcolor="#f8fafc", fontsize=9)
    ax.set_title(f"{player.iloc[0]['player_name']} | Headed Clearance Style Clusters", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_decision_profile(player_row: pd.Series, league: pd.DataFrame, out_path: pathlib.Path) -> None:
    metrics = [
        ("relief_oe", "Relief OE"),
        ("shot_prevention_oe", "Shot Prevent. OE"),
        ("decision_quality_oe", "Decision Quality"),
        ("length_oe", "Length OE"),
        ("territory_oe", "Territory OE"),
        ("wide_oe", "Wide OE"),
    ]
    values = [float(player_row[k]) for k, _ in metrics]
    league_values = [float(league[k].mean()) for k, _ in metrics]
    labels = [label for _, label in metrics]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    ax.barh(y + 0.18, league_values, height=0.32, color="#405160", label="League player average")
    ax.barh(y - 0.18, values, height=0.32, color="#ffbf69", label=str(player_row["player_name"]))
    ax.axvline(0, color="#d7dde2", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fafc")
    ax.tick_params(axis="x", colors="#d7dde2")
    ax.grid(axis="x", color="#405160", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.legend(frameon=False, labelcolor="#f8fafc", loc="lower right")
    ax.set_title(f"{player_row['player_name']} | Outcome Model Profile", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_player_timeline(player: pd.DataFrame, out_path: pathlib.Path) -> None:
    chart = player.sort_values(["match_id", "period_id", "elapsed_seconds"]).copy()
    chart["event_no"] = np.arange(1, len(chart) + 1)
    fig, ax = plt.subplots(figsize=(12, 5.8), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    colors = np.where(chart["decision_quality_oe"] >= 0, "#7ee081", "#ff6b6b")
    ax.bar(chart["event_no"], chart["decision_quality_oe"], color=colors, alpha=0.82, label="Decision quality OE")
    ax.plot(chart["event_no"], chart["p_relief_success"], color="#ffbf69", lw=2.0, marker="o", ms=4, label="P(relief success)")
    ax.plot(chart["event_no"], chart["p_opponent_shot_10s"], color="#7aa7ff", lw=2.0, marker="o", ms=4, label="P(opp. shot 10s)")
    ax.axhline(0, color="#d7dde2", lw=1.0, alpha=0.75)
    ax.set_title(f"{chart.iloc[0]['player_name']} | Decision Quality Timeline", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Headed clearance number", color="#d7dde2")
    ax.set_ylabel("Score / probability", color="#d7dde2")
    ax.tick_params(colors="#d7dde2")
    ax.grid(color="#405160", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.legend(frameon=False, labelcolor="#f8fafc", loc="upper right", ncol=3)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_rankings(player_summary: pd.DataFrame, out_path: pathlib.Path, min_clearances: int = 10) -> None:
    ranked = player_summary[player_summary["headed_clearances"] >= min_clearances].copy()
    ranked = ranked.sort_values("decision_quality_oe", ascending=False)
    plot_df = pd.concat([ranked.tail(12).sort_values("decision_quality_oe"), ranked.head(12)], ignore_index=True)
    labels = [f"{r.player_name} ({int(r.headed_clearances)})" for r in plot_df.itertuples()]
    colors = np.where(plot_df["decision_quality_oe"] >= 0, "#7ee081", "#ff6b6b")
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0b1117")
    ax.set_facecolor("#101820")
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["decision_quality_oe"], color=colors, alpha=0.85)
    ax.axvline(0, color="#d7dde2", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#f8fafc", fontsize=9)
    ax.tick_params(axis="x", colors="#d7dde2")
    ax.grid(axis="x", color="#405160", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#405160")
    ax.set_xlabel("Decision quality over expected per headed clearance", color="#d7dde2")
    ax.set_title(f"Outcome Model Rankings | Top and Bottom Players, min {min_clearances}", color="#f8fafc", fontsize=17, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_all_visuals(predictions: pd.DataFrame, player_summary: pd.DataFrame, player_name: str, out_dir: pathlib.Path) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    player = predictions[predictions["player_name"].str.casefold() == player_name.casefold()].copy()
    if player.empty:
        raise ValueError(f"No headed clearances found for {player_name}")
    player_row = player_summary[player_summary["player_name"].str.casefold() == player_name.casefold()].iloc[0]
    league = player_summary[player_summary["headed_clearances"] >= 5]
    prefix = safe_name(player.iloc[0]["player_name"])
    paths = [
        out_dir / f"{prefix}_decision_quality_map.png",
        out_dir / f"{prefix}_shot_risk_map.png",
        out_dir / f"{prefix}_style_cluster_map.png",
        out_dir / f"{prefix}_outcome_model_profile.png",
        out_dir / f"{prefix}_decision_timeline.png",
        out_dir / "relief_success_calibration.png",
        out_dir / "opponent_shot_calibration.png",
        out_dir / "decision_quality_rankings.png",
    ]
    save_player_outcome_map(player, paths[0])
    save_player_shot_risk_map(player, paths[1])
    save_player_style_map(player, paths[2])
    save_player_decision_profile(player_row, league, paths[3])
    save_player_timeline(player, paths[4])
    save_calibration_plot(predictions[predictions["is_test"]], "relief_success", "p_relief_success", "Relief Success Model Calibration", paths[5])
    save_calibration_plot(predictions[predictions["is_test"]], "opponent_shot_10s", "p_opponent_shot_10s", "Opponent Shot Within 10s Model Calibration", paths[6])
    save_rankings(player_summary, paths[7])
    return paths


def save_readme(out_path: pathlib.Path) -> None:
    out_path.write_text(
        """# Headed Clearance Outcome Model

This folder adds a decision-quality layer to the headed-clearance landing model.

## What is modeled

- `p_relief_success`: probability the clearance produces relief, based on the situation before the header.
- `p_opponent_shot_10s`: probability the opponent produces a shot within 10 seconds.
- `relief_oe`: actual relief outcome minus expected relief probability.
- `shot_prevention_oe`: expected shot risk minus actual shot outcome. Positive means the player avoided a shot that the model considered plausible.
- `decision_quality_oe`: composite decision score combining the existing landing-value score with relief over expected and shot prevention over expected.

## Important interpretation

These are analysis scores, not scouting absolutes. Use them to find clips and patterns:

- who turns bad situations into relief,
- who clears to areas with low second-wave risk,
- who gets same-team first touch more than expected,
- which clearance styles lead to danger.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.out_dir / "visuals"

    if not INSIGHTS_PATH.exists():
        raise FileNotFoundError(f"Run build_header_clearance_insights.py first: {INSIGHTS_PATH}")

    insights = pd.read_csv(INSIGHTS_PATH)
    events_by_match = load_match_events(args.data_dir)
    dataset = add_context(insights, events_by_match)
    bundle, predictions, metrics = train_outcome_models(dataset, args.random_state)
    predictions, style_summary, style_model = add_style_clusters(predictions, args.random_state)
    player_summary = summarize_players(predictions)

    dataset_path = args.out_dir / "headed_clearance_outcome_dataset.csv"
    predictions_path = args.out_dir / "headed_clearance_outcome_predictions.csv"
    player_summary_path = args.out_dir / "headed_clearance_player_decision_summary.csv"
    style_summary_path = args.out_dir / "headed_clearance_style_summary.csv"
    model_path = args.out_dir / "headed_clearance_outcome_models.joblib"
    metrics_path = args.out_dir / "outcome_model_metrics.json"
    readme_path = args.out_dir / "README.md"

    dataset.to_csv(dataset_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    player_summary.to_csv(player_summary_path, index=False)
    style_summary.to_csv(style_summary_path, index=False)
    joblib.dump({**bundle, "style_model": style_model}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_readme(readme_path)
    visual_paths = save_all_visuals(predictions, player_summary, args.player, vis_dir)

    for path in [
        dataset_path,
        predictions_path,
        player_summary_path,
        style_summary_path,
        model_path,
        metrics_path,
        readme_path,
        *visual_paths,
    ]:
        print(path)


if __name__ == "__main__":
    main()
