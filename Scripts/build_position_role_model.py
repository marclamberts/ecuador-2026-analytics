"""
Positional Role & Off-Ball Model for the Ecuador 2026 dataset.

Clusters outfield players into tactical role archetypes using:
  - average on-pitch position for defensive actions, passing, receiving,
    and shooting (where a player actually does things)
  - touch-zone occupancy shares and positional mobility (how much they roam)
  - off-ball involvement proxies: reception volume, progressive runs,
    pressing height, aerial presence, box presence -- signals that are not
    captured by raw goal/assist production

Broad position (GK/DEF/MID/FWD) is inferred from formation qualifiers, then
KMeans finds role sub-clusters within each outfield group. Clusters are
auto-labelled from their most distinguishing features.

Usage: python3 build_position_role_model.py
Outputs (in ../PositionRoleModel/):
  player_role_features.csv     -- full engineered feature matrix
  player_role_assignments.csv  -- player, position, cluster, role label
  role_cluster_profiles.csv    -- centroid profile per cluster
  role_model.joblib            -- {position: {scaler, kmeans, features}}
  model_meta.json              -- run metadata
  visuals/role_clusters_<POS>.png
  visuals/role_cluster_profiles.png
"""
import glob
import json
import os
import collections
import datetime
import json as json_mod

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"
AGG_DIR = os.path.join(DATA_DIR, "Aggregated")
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "PositionRoleModel")
VIS_DIR = os.path.join(OUT_DIR, "visuals")

MIN_MINUTES = 450
POS_LABEL = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD"}
DEF_ACTION_TYPES = {7, 8, 12, 44}
SHOT_TYPES = {13, 14, 15, 16}
TOUCH_TYPES = {1, 2, 3, 7, 8, 12, 13, 14, 15, 16, 42, 44, 49, 50, 61}
PENALTY_QID = 9

BG = "#0d1117"
C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
CLUSTER_COLORS = [C_NAVY, C_PINK, C_AMBER, C_PURPLE, C_INDIGO, C_CORAL]


def add_logo(fig, width=0.13, margin=0.014):
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


# ------------------------------------------------------- position inference --

def infer_positions(files):
    positions = collections.defaultdict(collections.Counter)
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e["typeId"] != 34:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            ids = (qmap.get(30) or "").split(", ")
            codes = (qmap.get(44) or "").split(", ")
            for pid, code in zip(ids, codes):
                if code in POS_LABEL:
                    positions[pid][code] += 1
    return {pid: POS_LABEL[counter.most_common(1)[0][0]] for pid, counter in positions.items()}


# ------------------------------------------------------ spatial aggregation --

def new_accum():
    return {"sum_x": 0.0, "sum_y": 0.0, "n": 0}


def add_point(acc, x, y):
    acc["sum_x"] += x
    acc["sum_y"] += y
    acc["n"] += 1


def collect_spatial_features(files):
    """One pass over every match, accumulating per-player_id position sums
    for defensive actions, passes, receptions, shots, and all touches
    (the latter also tracks sum of squares for a mobility/dispersion std)."""
    defn = collections.defaultdict(new_accum)
    passn = collections.defaultdict(new_accum)
    recvn = collections.defaultdict(new_accum)
    shotn = collections.defaultdict(new_accum)
    touch = collections.defaultdict(lambda: {"sum_x": 0.0, "sum_y": 0.0,
                                              "sum_x2": 0.0, "sum_y2": 0.0, "n": 0})

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        for i, e in enumerate(events):
            pid = e.get("playerId")
            tid = e["typeId"]
            x, y = e.get("x"), e.get("y")
            if pid and x is not None and y is not None:
                if tid in DEF_ACTION_TYPES:
                    add_point(defn[pid], x, y)
                if tid == 1:
                    add_point(passn[pid], x, y)
                if tid in SHOT_TYPES:
                    qids = set(q["qualifierId"] for q in e["qualifier"])
                    if PENALTY_QID not in qids:
                        add_point(shotn[pid], x, y)
                if tid in TOUCH_TYPES:
                    t = touch[pid]
                    t["sum_x"] += x
                    t["sum_y"] += y
                    t["sum_x2"] += x * x
                    t["sum_y2"] += y * y
                    t["n"] += 1

            # reception heuristic: successful pass, next same-team event by a
            # different player = that player received the ball there
            if tid == 1 and e.get("outcome") == 1 and i + 1 < len(events):
                nxt = events[i + 1]
                npid = nxt.get("playerId")
                if npid and npid != pid and nxt.get("contestantId") == e.get("contestantId"):
                    qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
                    ex = float(qmap.get(140, e["x"]))
                    ey = float(qmap.get(141, e["y"]))
                    add_point(recvn[npid], ex, ey)

    rows = []
    all_pids = set(defn) | set(passn) | set(recvn) | set(shotn) | set(touch)
    for pid in all_pids:
        row = {"player_id": pid}
        for label, acc in (("def", defn[pid]), ("pass", passn[pid]),
                           ("recv", recvn[pid]), ("shot", shotn[pid])):
            n = acc["n"]
            row[f"avg_{label}_x"] = acc["sum_x"] / n if n else np.nan
            row[f"avg_{label}_y"] = acc["sum_y"] / n if n else np.nan
            row[f"n_{label}"] = n
        t = touch[pid]
        n = t["n"]
        if n:
            mx, my = t["sum_x"] / n, t["sum_y"] / n
            row["avg_touch_x"] = mx
            row["avg_touch_y"] = my
            row["touch_std_x"] = max(t["sum_x2"] / n - mx ** 2, 0) ** 0.5
            row["touch_std_y"] = max(t["sum_y2"] / n - my ** 2, 0) ** 0.5
        else:
            row["avg_touch_x"] = row["avg_touch_y"] = np.nan
            row["touch_std_x"] = row["touch_std_y"] = np.nan
        row["n_touch"] = n
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------- off-ball features --

RAW_SUM_COLS = [
    "minutes", "matches", "touches", "central_touches", "halfspace_touches",
    "wide_touches", "box_touches", "final3_touches", "deep_touches",
    "defensive_third_def_actions", "middle_third_def_actions", "final_third_def_actions",
    "def_actions", "received_passes", "received_long_passes", "deep_completions",
    "progressive_runs", "accelerations", "progressive_carries", "carries",
    "aerials", "aerials_won", "territory_added", "field_tilt_touch", "crosses",
    "take_ons", "successful_take_ons", "tackles", "interceptions", "clearances",
    "recoveries", "shots", "goals", "xg", "key_passes", "xa",
    "progressive_passes", "passes", "completed_passes",
]


def build_offball_table():
    df = pd.read_csv(os.path.join(AGG_DIR, "player_season_metrics.csv"))
    grouped = df.groupby(["player_id", "player"], as_index=False)[RAW_SUM_COLS].sum()
    grouped["team"] = grouped["player_id"].map(
        df.sort_values("minutes", ascending=False).drop_duplicates("player_id")
          .set_index("player_id")["team"]
    )

    n90 = grouped["minutes"].replace(0, np.nan) / 90
    touches = grouped["touches"].replace(0, np.nan)
    def_actions = grouped["def_actions"].replace(0, np.nan)

    grouped["central_touch_share"] = grouped["central_touches"] / touches
    grouped["wide_touch_share"] = grouped["wide_touches"] / touches
    grouped["halfspace_touch_share"] = grouped["halfspace_touches"] / touches
    grouped["box_touch_share"] = grouped["box_touches"] / touches
    grouped["final3_touch_share"] = grouped["final3_touches"] / touches
    grouped["deep_touch_share"] = grouped["deep_touches"] / touches

    grouped["def_third_action_share"] = grouped["defensive_third_def_actions"] / def_actions
    grouped["mid_third_action_share"] = grouped["middle_third_def_actions"] / def_actions
    grouped["high_def_action_share"] = grouped["final_third_def_actions"] / def_actions

    grouped["received_passes_p90"] = grouped["received_passes"] / n90
    grouped["received_long_passes_p90"] = grouped["received_long_passes"] / n90
    grouped["deep_completions_p90"] = grouped["deep_completions"] / n90
    grouped["progressive_runs_p90"] = grouped["progressive_runs"] / n90
    grouped["accelerations_p90"] = grouped["accelerations"] / n90
    grouped["progressive_carries_p90"] = grouped["progressive_carries"] / n90
    grouped["carries_p90"] = grouped["carries"] / n90
    grouped["aerials_p90"] = grouped["aerials"] / n90
    grouped["aerial_win_pct"] = grouped["aerials_won"] / grouped["aerials"].replace(0, np.nan)
    grouped["territory_added_p90"] = grouped["territory_added"] / n90
    grouped["field_tilt_touch_p90"] = grouped["field_tilt_touch"] / n90
    grouped["crosses_p90"] = grouped["crosses"] / n90
    grouped["take_ons_p90"] = grouped["take_ons"] / n90
    grouped["take_on_pct"] = grouped["successful_take_ons"] / grouped["take_ons"].replace(0, np.nan)
    grouped["progressive_passes_p90"] = grouped["progressive_passes"] / n90
    grouped["def_actions_p90"] = grouped["def_actions"] / n90
    grouped["box_touches_p90"] = grouped["box_touches"] / n90

    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    return grouped


# --------------------------------------------------------------- clustering --

# feature -> (high_phrase, low_phrase or None) used for auto-labelling
FEATURE_TAGS = {
    "press_height": ("High Press", "Deep Block"),
    "pass_width": ("Wide Passer", "Central Passer"),
    "recv_width": ("Wide Receiver", "Central Receiver"),
    "def_width": ("Wide Defender", "Central Defender"),
    "touch_height": ("Advanced Positioning", "Deep Positioning"),
    "touch_mobility": ("Roaming / Mobile", "Positionally Disciplined"),
    "central_touch_share": ("Central Focus", None),
    "wide_touch_share": ("Wide Focus", None),
    "halfspace_touch_share": ("Halfspace Focus", None),
    "box_touch_share": ("Box Presence", None),
    "final3_touch_share": ("Final-Third Presence", None),
    "deep_touch_share": ("Deep Buildup Role", None),
    "high_def_action_share": ("High Press Actions", "Low Block Actions"),
    "received_passes_p90": ("High-Volume Receiver", None),
    "progressive_runs_p90": ("Ball-Carrying Runner", None),
    "aerial_win_pct": ("Aerial Presence", None),
    "crosses_p90": ("Crosser", None),
    "take_ons_p90": ("Dribbler", None),
    "progressive_carries_p90": ("Carrier", None),
    "progressive_passes_p90": ("Line-Breaking Passer", None),
    "def_actions_p90": ("High Defensive Workrate", None),
}

CLUSTER_FEATURES = list(FEATURE_TAGS.keys())


def engineer_features(spatial, offball, pos_map):
    df = offball.merge(spatial, on="player_id", how="inner")
    df["position"] = df["player_id"].map(pos_map)
    df = df[df["minutes"] >= MIN_MINUTES].copy()

    df["press_height"] = df["avg_def_x"]
    df["pass_width"] = (df["avg_pass_y"] - 50).abs()
    df["recv_width"] = (df["avg_recv_y"] - 50).abs()
    df["def_width"] = (df["avg_def_y"] - 50).abs()
    df["touch_height"] = df["avg_touch_x"]
    df["touch_mobility"] = df["touch_std_x"] + df["touch_std_y"]

    return df


def label_cluster(centroid_z, used_features, max_tags=3):
    ranked = sorted(used_features, key=lambda f: -abs(centroid_z[f]))
    tags = []
    for f in ranked:
        if len(tags) == max_tags:
            break
        high, low = FEATURE_TAGS[f]
        z = centroid_z[f]
        if z >= 0:
            tags.append(high)
        elif low is not None:
            tags.append(low)
        else:
            continue
    return " + ".join(tags) if tags else "Balanced"


# Role granularity is a deliberate choice, not a silhouette-maximizing one:
# silhouette on this feature space keeps favoring k=2 (a coarse wide/central
# split) all the way out, since real playing styles form a continuum rather
# than tight, well-separated blobs. A fixed k per group -- capped by group
# size -- gives tactically legible sub-roles instead of always collapsing to
# the least informative split. Silhouette is still recorded as a diagnostic.
FIXED_K = {"DEF": 4, "MID": 4, "FWD": 3}


def fit_position_group(df_pos, features, random_state=42):
    X = df_pos[features].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n = len(df_pos)
    position = df_pos["position"].iloc[0]
    k = min(FIXED_K.get(position, 4), max(2, n // 10))
    best_km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(Xs)
    best_labels = best_km.labels_
    best_k = k
    best_score = silhouette_score(Xs, best_labels) if len(set(best_labels)) > 1 else float("nan")

    profiles = {}
    z_df = pd.DataFrame(Xs, columns=features, index=df_pos.index)
    for c in range(best_k):
        mask = best_labels == c
        centroid_z = z_df[mask].mean().to_dict()
        profiles[c] = {
            "n_players": int(mask.sum()),
            "centroid_z": centroid_z,
            "centroid_raw": X[mask].mean().to_dict(),
            "label": label_cluster(centroid_z, features),
        }

    return {
        "scaler": scaler, "kmeans": best_km, "features": features,
        "k": best_k, "silhouette": best_score,
    }, best_labels, profiles


# -------------------------------------------------------------------- plot --

def plot_position_clusters(df_pos, labels, profiles, position, out_path):
    pca = PCA(n_components=2, random_state=42)
    X = StandardScaler().fit_transform(df_pos[CLUSTER_FEATURES].fillna(df_pos[CLUSTER_FEATURES].median()))
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for c in sorted(profiles):
        mask = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=70, color=color, alpha=0.8,
                   edgecolors="white", linewidths=0.6,
                   label=f"{profiles[c]['label']} (n={profiles[c]['n_players']})")

    top = df_pos.sort_values("minutes", ascending=False).head(14)
    for idx in top.index:
        pos_idx = df_pos.index.get_loc(idx)
        ax.annotate(str(df_pos.loc[idx, "player"]).split()[-1],
                    (coords[pos_idx, 0], coords[pos_idx, 1]),
                    fontsize=7.5, color="#c7ccd4", alpha=0.9,
                    xytext=(4, 3), textcoords="offset points")

    ax.set_title(f"{position} — Positional Role Clusters", fontsize=15, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("PCA 1", color="#9aa4b2", fontsize=9)
    ax.set_ylabel("PCA 2", color="#9aa4b2", fontsize=9)
    ax.tick_params(colors="#6b7684", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2c3540")
    ax.legend(loc="best", frameon=False, fontsize=8.5, labelcolor="#e6e9ed")
    fig.text(0.01, 0.01, "Data via Opta | Ecuador 2026 event data · KMeans on standardized off-ball/positional features",
              fontsize=7.5, color="#6b7684")
    add_logo(fig)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def plot_cluster_profile_heatmap(all_profiles, out_path):
    rows, row_labels = [], []
    for position, profiles in all_profiles.items():
        for c in sorted(profiles):
            rows.append(profiles[c]["centroid_z"])
            row_labels.append(f"{position}: {profiles[c]['label']} (n={profiles[c]['n_players']})")

    mat = pd.DataFrame(rows, columns=CLUSTER_FEATURES).values
    fig, ax = plt.subplots(figsize=(14, max(4, 0.5 * len(rows) + 1.5)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(CLUSTER_FEATURES)))
    ax.set_xticklabels(CLUSTER_FEATURES, rotation=60, ha="right", fontsize=8, color="#c7ccd4")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, color="#e6e9ed")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("z-score vs. position group", color="#c7ccd4", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#6b7684", labelcolor="#c7ccd4")
    ax.set_title("Role Cluster Profiles — Standardized Feature Signature", fontsize=14,
                 fontweight="bold", color="white", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


# ------------------------------------------------------------------- main --

def main():
    os.makedirs(VIS_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    print(f"Found {len(files)} match files")

    pos_map = infer_positions(files)
    spatial = collect_spatial_features(files)
    offball = build_offball_table()
    df = engineer_features(spatial, offball, pos_map)
    df = df[df["position"].isin(["DEF", "MID", "FWD"])].reset_index(drop=True)
    print(f"{len(df)} outfield players with >= {MIN_MINUTES} minutes")

    models, assignments, all_profiles = {}, [], {}
    for position in ["DEF", "MID", "FWD"]:
        df_pos = df[df["position"] == position].reset_index(drop=True)
        if len(df_pos) < 6:
            print(f"Skipping {position}: only {len(df_pos)} players")
            continue
        model, labels, profiles = fit_position_group(df_pos, CLUSTER_FEATURES)
        models[position] = model
        all_profiles[position] = profiles
        df_pos["cluster"] = labels
        df_pos["role_label"] = df_pos["cluster"].map(lambda c: profiles[c]["label"])
        assignments.append(df_pos)
        print(f"{position}: k={model['k']} silhouette={model['silhouette']:.3f}")
        plot_position_clusters(df_pos, labels, profiles, position,
                                os.path.join(VIS_DIR, f"role_clusters_{position}.png"))

    assigned = pd.concat(assignments, ignore_index=True)
    feature_cols = ["player_id", "player", "team", "position", "minutes", "matches"] + CLUSTER_FEATURES
    assigned[feature_cols + ["n_def", "n_pass", "n_recv", "n_shot"]].to_csv(
        os.path.join(OUT_DIR, "player_role_features.csv"), index=False)

    assign_cols = ["player_id", "player", "team", "position", "minutes", "matches", "cluster", "role_label"]
    assigned[assign_cols].sort_values(["position", "cluster", "minutes"], ascending=[True, True, False]).to_csv(
        os.path.join(OUT_DIR, "player_role_assignments.csv"), index=False)

    profile_rows = []
    for position, profiles in all_profiles.items():
        for c, p in profiles.items():
            row = {"position": position, "cluster": c, "role_label": p["label"], "n_players": p["n_players"]}
            row.update({f"{k}_z": v for k, v in p["centroid_z"].items()})
            row.update({f"{k}_raw": v for k, v in p["centroid_raw"].items()})
            profile_rows.append(row)
    pd.DataFrame(profile_rows).to_csv(os.path.join(OUT_DIR, "role_cluster_profiles.csv"), index=False)

    joblib.dump(models, os.path.join(OUT_DIR, "role_model.joblib"))

    meta = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "min_minutes": MIN_MINUTES,
        "n_players": len(assigned),
        "features": CLUSTER_FEATURES,
        "groups": {p: {"k": m["k"], "silhouette": m["silhouette"], "n_players": len(assignments[i])}
                   for i, (p, m) in enumerate(models.items())},
    }
    with open(os.path.join(OUT_DIR, "model_meta.json"), "w") as f:
        json_mod.dump(meta, f, indent=2)

    plot_cluster_profile_heatmap(all_profiles, os.path.join(VIS_DIR, "role_cluster_profiles.png"))

    print("\nDone. Outputs written to", OUT_DIR)


if __name__ == "__main__":
    main()
