"""
Season pass network for a team, split into First Half / Second Half panels,
with players grouped into their most common pass-combination clusters
(weighted greedy-modularity community detection on completed-pass counts).
Nodes/edges/hulls are coloured by cluster so the recurring passing units
(e.g. back line, double pivot, front group) stand out.

Usage: python3 pass_network_clusters.py "Independiente del Valle" [out.png]
"""
import glob
import json
import re
import sys
import collections

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
from mplsoccer import VerticalPitch
from adjustText import adjust_text

DATA_DIR = "/Users/marclamberts/Event data/Ecuador 2026"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
BG = "#0d1117"
PITCH_LINE = "#2c3a4d"
EDGE_COLOR = "#c7ccd4"

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"


def add_logo(fig, width=0.175, margin=0.018):
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


CORE_SQUAD_SIZE = 14  # cap node count to keep the network readable
MIN_EDGE_PASSES = 10  # minimum combined passes between a pair to draw a line
PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")

CLUSTER_COLORS = [C_AMBER, C_PINK, C_NAVY, C_CORAL, C_PURPLE, C_INDIGO]


def detect_clusters(edges_half, node_ids):
    """Group players into pass-combination clusters via greedy modularity
    community detection on the weighted (pass-count) graph."""
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for (a, b), cnt in edges_half.items():
        if a in node_ids and b in node_ids:
            graph.add_edge(a, b, weight=cnt)
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    cluster_of = {}
    for idx, comm in enumerate(communities):
        for pid in comm:
            cluster_of[pid] = idx
    return cluster_of


def clean_name(name):
    return PREFIX_RE.sub("", name)


def build_team_map(files):
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def clock_to_elapsed(period_id, time_min, p1_len, p2_len):
    if period_id == 1:
        return min(time_min, p1_len)
    if period_id == 2:
        return p1_len + max(0, time_min - 45)
    return p1_len + p2_len


def compute_minutes(files, cid):
    minutes = collections.defaultdict(float)
    names = {}
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = data["event"]
        if not any(e.get("contestantId") == cid for e in events):
            continue
        periods = data["matchDetails"]["period"]
        p1_len = next((p["lengthMin"] for p in periods if p["id"] == 1), 45)
        p2_len = next((p["lengthMin"] for p in periods if p["id"] == 2), 45)
        match_total = p1_len + p2_len

        starters = {}
        for e in events:
            if e["typeId"] != 34 or e.get("contestantId") != cid:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            ids = (qmap.get(30) or "").split(", ")
            codes = (qmap.get(44) or "").split(", ")
            for pid, code in zip(ids, codes):
                if code in ("1", "2", "3", "4"):
                    starters[pid] = 0.0

        on_time = dict(starters)
        off_time = {}
        for e in events:
            if e.get("contestantId") != cid:
                continue
            if e["typeId"] == 18:
                pid = e.get("playerId")
                if pid:
                    off_time[pid] = clock_to_elapsed(e["periodId"], e["timeMin"], p1_len, p2_len)
            elif e["typeId"] == 19:
                pid = e.get("playerId")
                if pid:
                    on_time[pid] = clock_to_elapsed(e["periodId"], e["timeMin"], p1_len, p2_len)
            if e.get("playerId") and e.get("playerName"):
                names[e["playerId"]] = e["playerName"]

        for pid, t_on in on_time.items():
            t_off = off_time.get(pid, match_total)
            minutes[pid] += max(0.0, t_off - t_on)

    return minutes, names


def formation_snapshots(data, cid):
    """Chronological list of (period, timeMin, position-codes) for a team,
    built from TeamSetup (34, pre-match) and FormationChange (40) events.
    Pre-match TeamSetup carries periodId 16 in this feed; normalised to 0
    so it always sorts first."""
    snaps = []
    for e in data["event"]:
        if e["typeId"] not in (34, 40) or e.get("contestantId") != cid:
            continue
        qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
        codes = (qmap.get(44) or "").split(", ")
        if not codes:
            continue
        period = e.get("periodId", 16)
        period = 0 if period == 16 else period
        snaps.append((period, e.get("timeMin", 0), codes))
    snaps.sort(key=lambda s: (s[0], s[1]))
    return snaps


def formation_string(codes):
    on_pitch = [c for c in codes if c in ("1", "2", "3", "4")]
    d, m, f = on_pitch.count("2"), on_pitch.count("3"), on_pitch.count("4")
    if d + m + f == 0:
        return None
    return f"{d}-{m}-{f}"


def formation_for_half(snaps, half):
    if half == 1:
        candidates = [s for s in snaps if s[0] == 0]
    else:
        candidates = [s for s in snaps if s[0] in (0, 1)]
    if not candidates:
        candidates = snaps[:1]
    if not candidates:
        return None
    return formation_string(candidates[-1][2])


def most_common_formations(files, cid):
    counts = {1: collections.Counter(), 2: collections.Counter()}
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        if not any(e.get("contestantId") == cid for e in data["event"]):
            continue
        snaps = formation_snapshots(data, cid)
        if not snaps:
            continue
        for half in (1, 2):
            shape = formation_for_half(snaps, half)
            if shape:
                counts[half][shape] += 1
    result = {}
    for half in (1, 2):
        result[half] = counts[half].most_common(1)[0][0] if counts[half] else None
    return result


def collect_network(files, cid, core_players):
    # half -> player -> [x,y,...] positions ; half -> (a,b) -> count
    positions = {1: collections.defaultdict(list), 2: collections.defaultdict(list)}
    edges = {1: collections.Counter(), 2: collections.Counter()}
    pass_counts = {1: collections.Counter(), 2: collections.Counter()}

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        evs.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

        for i, e in enumerate(evs):
            if e["typeId"] != 1 or e.get("contestantId") != cid:
                continue
            pid = e.get("playerId")
            if pid not in core_players:
                continue
            half = e["periodId"]
            positions[half][pid].append((e["x"], e["y"]))
            pass_counts[half][pid] += 1

            if e["outcome"] != 1:
                continue
            if i + 1 >= len(evs):
                continue
            nxt = evs[i + 1]
            if nxt.get("contestantId") != cid:
                continue
            rpid = nxt.get("playerId")
            if rpid is None or rpid == pid or rpid not in core_players:
                continue
            qmap = {q["qualifierId"]: q.get("value") for q in e["qualifier"]}
            ex = float(qmap.get(140, e["x"]))
            ey = float(qmap.get(141, e["y"]))
            positions[half][rpid].append((ex, ey))
            pair = tuple(sorted((pid, rpid)))
            edges[half][pair] += 1

    return positions, edges, pass_counts


def make_plot(team_name, files, cid, out_path):
    minutes, names = compute_minutes(files, cid)
    ranked = sorted(minutes.items(), key=lambda kv: -kv[1])[:CORE_SQUAD_SIZE]
    core_players = {pid for pid, m in ranked}
    min_minutes_used = min(m for _, m in ranked)
    if not core_players:
        raise SystemExit("No players met the minutes threshold.")

    positions, edges, pass_counts = collect_network(files, cid, core_players)
    formations = most_common_formations(files, cid)

    display_name = clean_name(team_name)
    fig = plt.figure(figsize=(17, 11.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.03, 0.965, display_name, fontsize=28, fontweight="bold", color="white")
    fig.text(0.03, 0.930, "Season Pass Network · Most Common Pass Clusters", fontsize=14.5,
             fontweight="bold", color="#c7ccd4")
    fig.text(0.03, 0.902, f"Ecuador 2026 · All Competitions · top {len(core_players)} players by minutes "
             f"(≥{min_minutes_used:.0f} min) · node size = passes, line width = combinations · "
             f"colour = most common pass cluster",
             fontsize=10.5, color="#9aa4b2")

    panel_titles = {1: "FIRST HALF", 2: "SECOND HALF"}
    axes = [fig.add_axes([0.03 + i * 0.48, 0.05, 0.44, 0.82]) for i in range(2)]

    for ax, half in zip(axes, (1, 2)):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                               linewidth=1.2, half=False)
        pitch.draw(ax=ax)
        shape = formations.get(half)
        title = panel_titles[half]
        if shape:
            title = f"{panel_titles[half]}   ·   MOST USED FORMATION: {shape}"
        ax.set_title(title, fontsize=13, fontweight="bold", color="#e6e9ee", pad=10)

        avg_pos = {}
        for pid in core_players:
            pts = positions[half].get(pid, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            avg_pos[pid] = (sum(xs) / len(xs), sum(ys) / len(ys))

        max_passes = max(pass_counts[half].values(), default=1)
        max_edge = max(edges[half].values(), default=1)

        cluster_of = detect_clusters(edges[half], set(avg_pos))
        cluster_members = collections.defaultdict(list)
        for pid in avg_pos:
            cluster_members[cluster_of.get(pid, -1)].append(pid)

        # shaded hull behind each cluster of 3+ players, coloured by cluster
        for cl_idx, members in cluster_members.items():
            if len(members) < 3:
                continue
            color = CLUSTER_COLORS[cl_idx % len(CLUSTER_COLORS)]
            xs = [avg_pos[p][0] for p in members]
            ys = [avg_pos[p][1] for p in members]
            hull = pitch.convexhull(xs, ys)
            pitch.polygon(hull, ax=ax, facecolor=color, edgecolor=color,
                         alpha=0.14, lw=1.2, zorder=1)

        for (a, b), cnt in edges[half].items():
            if cnt < MIN_EDGE_PASSES or a not in avg_pos or b not in avg_pos:
                continue
            x0, y0 = avg_pos[a]
            x1, y1 = avg_pos[b]
            lw = 0.8 + (cnt / max_edge) * 6.5
            same_cluster = cluster_of.get(a) == cluster_of.get(b)
            color = CLUSTER_COLORS[cluster_of.get(a, 0) % len(CLUSTER_COLORS)] if same_cluster else EDGE_COLOR
            alpha = (0.35 + (cnt / max_edge) * 0.55) if same_cluster else (0.12 + (cnt / max_edge) * 0.2)
            pitch.lines(x0, y0, x1, y1, ax=ax, color=color, lw=lw, alpha=alpha,
                       zorder=2, comet=False)

        texts = []
        for pid, (x, y) in avg_pos.items():
            n_passes = pass_counts[half].get(pid, 0)
            size = 220 + (n_passes / max_passes) * 1300
            node_color = CLUSTER_COLORS[cluster_of.get(pid, 0) % len(CLUSTER_COLORS)]
            pitch.scatter(x, y, ax=ax, s=size, color=node_color, edgecolors=BG,
                          linewidths=2, zorder=3, alpha=0.95)
            label = names.get(pid, "?").split(" ")[-1]
            ann = pitch.annotate(
                label, xy=(x, y), ax=ax, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color="white", zorder=5, clip_on=False,
                path_effects=[pe.withStroke(linewidth=2.6, foreground=BG)],
                arrowprops=dict(arrowstyle="-", color="#8b93a1", lw=0.7, alpha=0.85,
                                 shrinkA=0, shrinkB=4),
            )
            texts.append(ann)

        adjust_text(texts, ax=ax, expand=(1.25, 1.35),
                    force_text=(0.4, 0.6), force_static=(0.3, 0.4),
                    only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"})

    add_logo(fig)
    fig.text(0.03, 0.015, "Data via Opta | Ecuador 2026 event data", fontsize=8.5, color="#6b7684")
    fig.text(0.97, 0.015, "Edge = completed pass between two core players, next-touch heuristic",
             fontsize=8, color="#6b7684", ha="right")

    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)
    for half in (1, 2):
        total_edges = sum(1 for (a, b), c in edges[half].items() if c >= MIN_EDGE_PASSES and a in core_players and b in core_players)
        print(f"Half {half}: players_with_data={sum(1 for p in core_players if positions[half].get(p))} "
              f"edges>={MIN_EDGE_PASSES}={total_edges} total_passes={sum(pass_counts[half].values())}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Independiente del Valle"
    out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/pass_network_clusters_{team_name.replace(' ', '_')}.png"

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]
    make_plot(match, files, cid, out)
