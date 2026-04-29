#!/usr/bin/env python3
"""
elermorevale_gui_v2.py — Improved dual-view power flow dashboard.

This is a refactor of elermorevale_gui.py with the following additions:

    1. Optional Flask backend for interactive day/scenario switching.
       Run with --serve to start the server, or omit for static HTML.
    2. Visible data-health badges (monitors mapped, TX monitor OK,
       solver converged) — silent failures from the original are now loud.
    3. Time cursor synchronised across network, substation power chart,
       voltage histogram, and the new violation timeline.
    4. Per-timestep violation timeline (under/over) shown for both
       baseline and QP, so users see at a glance where QP helps.
    5. Hierarchical layout option using BFS depth, with spring layout
       retained as a fallback. Backbone runs horizontally, LV trees
       branch vertically.
    6. Hover tooltip extended with distance-from-source, parent TX,
       baseline-vs-QP voltage delta, and per-day violation count.
    7. URL parameter state: day, scenario, hour, zoom, highlight, view.
    8. Performance: canvas batched draws unchanged; WebGL hook documented
       (deck.gl/regl) but not implemented to avoid extra dependencies.
    9. Export: "Save view as PNG" and "Export timestep CSV" buttons.
    10. Solver status surfaced in sidebar (converged / max-iter warning).

Usage:
    # Static HTML (one day, no backend)
    python elermorevale_gui_v2.py --simulate --day 190 --open

    # Interactive backend (any day, any scenario, on demand)
    python elermorevale_gui_v2.py --serve --port 8765

    # --layout spring (for organic layout)

The static path stays compatible with the original CLI, so existing
workflows keep working.
"""
import argparse
import glob
import json
import logging
import os
import re
import webbrowser
from collections import defaultdict, deque

import networkx as nx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==================================================================
# GLM TOPOLOGY PARSING (unchanged from v1)
# ==================================================================

def parse_glm(fp):
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)
    objs = []
    for m in re.finditer(r"object\s+(\w+)\s*\{([^}]*)\}", text, re.DOTALL):
        props = {}
        for pm in re.finditer(r"([\w.]+)\s+([^;]+);", m.group(2)):
            props[pm.group(1).strip()] = pm.group(2).strip()
        objs.append((m.group(1), props))
    return objs


def gfloat(v, d=0.0):
    if v is None:
        return float(d)
    s = str(v).strip()
    m = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else float(d)


def build_topology(glm_dir):
    """Parse all GLM files, return a NetworkX graph with attributes."""
    all_objs = []
    for fp in sorted(glob.glob(os.path.join(glm_dir, "**", "*.glm"),
                               recursive=True)):
        if "__MACOSX" in fp:
            continue
        for ot, p in parse_glm(fp):
            all_objs.append((ot, p))

    by_type = defaultdict(list)
    for ot, p in all_objs:
        by_type[ot].append(p)

    G = nx.Graph()
    bus_vl = {}
    parent_tx = {}  # (#6) which distribution TX a bus sits behind

    for p in by_type.get("transformer", []):
        f, t = p.get("from", ""), p.get("to", "")
        name = p.get("name", "")
        if f and t:
            G.add_edge(f, t, element="transformer", name=name)
            if "132" in f or "Jesmond" in f:
                bus_vl[f] = "HV"
                bus_vl[t] = "MV"
            elif "TXZoneSub" not in name:
                bus_vl.setdefault(f, "MV")
                bus_vl[t] = "LV"
                parent_tx[t] = name

    for p in by_type.get("regulator", []):
        f, t = p.get("from", ""), p.get("to", "")
        if f and t:
            G.add_edge(f, t, element="regulator", name=p.get("name", ""))

    for ot in ["overhead_line", "underground_line", "triplex_line"]:
        for p in by_type.get(ot, []):
            f, t = p.get("from", ""), p.get("to", "")
            length = gfloat(p.get("length", "1"), 1.0)
            if f and t:
                G.add_edge(f, t, element=ot, length=length)

    for ot in ["switch", "fuse"]:
        for p in by_type.get(ot, []):
            f, t = p.get("from", ""), p.get("to", "")
            if f and t:
                G.add_edge(f, t, element=ot, length=0.001)

    for n in G.nodes():
        if n not in bus_vl:
            if "132" in n or "Jesmond" in n:
                bus_vl[n] = "HV"
            elif n.startswith("_100") or "BusZone" in n:
                bus_vl[n] = "MV"
            else:
                bus_vl[n] = "LV"
    nx.set_node_attributes(G, bus_vl, "vl")

    parent_of = {}
    for pl in by_type.values():
        for p in pl:
            n, par = p.get("name", ""), p.get("parent", "")
            if n and par:
                parent_of[n] = par

    def resolve(nm):
        c = nm
        for _ in range(10):
            if c not in parent_of:
                return c
            c = parent_of[c]
        return c

    lc = defaultdict(int)
    for p in by_type.get("load", []):
        lc[resolve(p.get("parent", ""))] += 1

    nx.set_node_attributes(G, {n: n in lc for n in G.nodes()}, "hl")
    nx.set_node_attributes(G, dict(lc), "nl")

    # BFS depth + electrical distance for #6
    source = None
    for n in G.nodes():
        if "Jesmond" in n or "132" in n:
            source = n
            break

    depth, dist_m = {}, {}
    if source:
        depth[source] = 0
        dist_m[source] = 0.0
        q = deque([source])
        while q:
            c = q.popleft()
            for nb in G.neighbors(c):
                if nb not in depth:
                    depth[nb] = depth[c] + 1
                    edge_len = G.edges[c, nb].get("length", 1.0)
                    dist_m[nb] = dist_m[c] + edge_len
                    q.append(nb)

    nx.set_node_attributes(G, depth, "depth")
    nx.set_node_attributes(G, dist_m, "dist_m")

    # propagate parent_tx down LV trees
    full_parent_tx = {}
    if source:
        # initialise: nodes directly downstream of a TX inherit its name
        for bus, txn in parent_tx.items():
            full_parent_tx[bus] = txn
        # walk from source, inheriting from MV side as we cross TXs
        q = deque([source])
        seen = {source}
        while q:
            c = q.popleft()
            for nb in G.neighbors(c):
                if nb in seen:
                    continue
                seen.add(nb)
                if nb not in full_parent_tx and c in full_parent_tx:
                    full_parent_tx[nb] = full_parent_tx[c]
                q.append(nb)
    nx.set_node_attributes(G, full_parent_tx, "tx")

    logger.info("Topology: %d nodes, %d edges, %d components",
                G.number_of_nodes(), G.number_of_edges(),
                nx.number_connected_components(G))
    return G, source


# ==================================================================
# LAYOUT — hierarchical (preferred) or spring (fallback) — (#5)
# ==================================================================

def hierarchical_layout(G, source, width=2.4, height=1.8):
    """
    Tidy-tree layout: each subtree owns a vertical band proportional to
    its leaf count, so dense branches don't crush sparse ones. x runs
    by BFS depth from `source`. This gives a readable radial-feeder
    picture without spring layout's randomness.

    Falls back to spring_layout if there is no source or the graph has
    cycles that make a tree projection meaningless.
    """
    if source is None or source not in G:
        logger.warning("No source for hierarchical layout; using spring")
        return nx.spring_layout(G, k=2.0, iterations=80, seed=42)

    # Build a BFS tree rooted at the source so each non-root node has
    # exactly one parent. Edges not in the tree are still drawn — we're
    # only using the tree for placement, not topology.
    parent = {source: None}
    children = defaultdict(list)
    order = [source]
    visited = {source}
    queue = deque([source])
    while queue:
        c = queue.popleft()
        # sort neighbours for deterministic layout
        for nb in sorted(G.neighbors(c)):
            if nb in visited:
                continue
            visited.add(nb)
            parent[nb] = c
            children[c].append(nb)
            order.append(nb)
            queue.append(nb)

    # Disconnected components fall back to spring (shouldn't happen on
    # the Elermore Vale network, but be defensive)
    if len(visited) < G.number_of_nodes():
        unreached = G.number_of_nodes() - len(visited)
        logger.warning("Hierarchical layout: %d unreached nodes; "
                       "falling back to spring", unreached)
        return nx.spring_layout(G, k=2.0, iterations=80, seed=42)

    # Count leaves under each subtree (post-order traversal of BFS tree)
    leaves = {}
    for n in reversed(order):
        if not children[n]:
            leaves[n] = 1
        else:
            leaves[n] = sum(leaves[c] for c in children[n])

    # Walk the tree top-down assigning each node a y-band
    # proportional to its leaf count.
    pos = {}
    depths = nx.get_node_attributes(G, "depth")
    max_d = max(depths.values()) if depths else 1

    def place(node, y_start, y_end):
        # x from BFS depth
        d = depths.get(node, 0)
        x = (d / max_d) * width - width / 2
        # node sits at midpoint of its band
        pos[node] = (x, (y_start + y_end) / 2)
        # split band among children proportional to their leaf counts
        kids = children[node]
        if not kids:
            return
        total = leaves[node]
        cursor = y_start
        for k in kids:
            frac = leaves[k] / total
            band = (y_end - y_start) * frac
            place(k, cursor, cursor + band)
            cursor += band

    place(source, -height / 2, height / 2)

    # Tiny jitter so coincident points (e.g. a load's parent meter and
    # the load itself end up at exactly the same coords) separate visually
    rng = np.random.default_rng(42)
    for n in pos:
        jx, jy = rng.normal(0, 0.003, 2)
        pos[n] = (pos[n][0] + jx, pos[n][1] + jy)

    return pos


# ==================================================================
# SIMULATION (with health reporting — #2, #10)
# ==================================================================

def run_simulation(glm_dir, common_dir, profiles_csv, day_idx,
                   n_monitors=150):
    """
    Run baseline + QP for one day. Returns a dict including health
    indicators so the frontend can show whether the data is trustworthy.
    """
    from elermorevale_openDSS import (
        build_elermorevale, get_network_load_names,
        map_customers_to_network_loads, select_monitored_loads,
        add_monitors, attach_baseline_shapes, attach_loadshapes,
        run_daily, collect_voltages, collect_tx_power, collect_losses,
        load_profiles_from_csv,
    )
    from dss import DSS as dss

    health = {
        "profile_csv": profiles_csv,
        "day_idx": day_idx,
        "monitors_requested": n_monitors,
        "monitors_added": 0,
        "load_bus_map_size": 0,
        "tx_monitor_ok": False,
        "scenarios": {},
    }

    profiles = load_profiles_from_csv(profiles_csv)
    cids = sorted(profiles.keys())
    health["customers"] = len(cids)

    build_elermorevale(glm_dir, common_dir, skip_generators=True)
    ln = get_network_load_names()
    health["loads_in_circuit"] = len(ln)

    lcm = map_customers_to_network_loads(cids, ln)
    mon = select_monitored_loads(lcm, n_monitors=n_monitors)
    health["monitors_added"] = len(mon)

    res = {}
    for lbl, ub in [("baseline", True), ("qp", False)]:
        logger.info("Running %s for day %d ...", lbl, day_idx)
        scen_health = {"converged": None, "warnings": []}

        build_elermorevale(glm_dir, common_dir, skip_generators=True)
        add_monitors(mon)

        if ub:
            ds = attach_baseline_shapes(lcm, profiles, day_idx)
        else:
            ds = attach_loadshapes(lcm, profiles, day_idx)

        # Capture solver convergence (#10)
        try:
            run_daily()
            scen_health["converged"] = bool(
                dss.ActiveCircuit.Solution.Converged
            )
        except Exception as exc:
            msg = str(exc)
            scen_health["warnings"].append(msg)
            if "485" in msg:
                scen_health["converged"] = "max_iter_warning"
            else:
                scen_health["converged"] = False
                logger.error("Solver failed for %s: %s", lbl, msg)

        v = collect_voltages(mon)
        tp, tq = collect_tx_power()
        lk, _ = collect_losses()

        # TX monitor health: did we get nonzero data?
        scen_health["tx_monitor_nonzero"] = bool(np.any(np.abs(tp) > 0.01))
        if scen_health["tx_monitor_nonzero"]:
            health["tx_monitor_ok"] = True

        res[lbl] = {
            "voltages": {k: val.tolist() for k, val in v.items()},
            "tx_p": tp.tolist(),
            "tx_q": tq.tolist(),
            "loss_kw": float(lk),
            "date": ds,
            "health": scen_health,
        }

    # Build load → bus map with case-insensitive matching (the original bug)
    lbm = {}
    lbm_failures = 0
    for l in mon:
        try:
            dss.ActiveCircuit.SetActiveElement(f"Load.{l}")
            bn = dss.ActiveCircuit.ActiveElement.BusNames
            if bn:
                lbm[l] = bn[0].split(".")[0]
        except Exception as exc:
            lbm_failures += 1
            logger.warning("Bus lookup failed for Load.%s: %s", l, exc)

    health["load_bus_map_size"] = len(lbm)
    health["load_bus_map_failures"] = lbm_failures
    res["load_bus_map"] = lbm
    res["health"] = health
    return res


# ==================================================================
# DASHBOARD GENERATION
# ==================================================================

def _safe_node_lookup(node_list):
    """Build case-insensitive lookup so we never silently drop monitors."""
    return {n.lower(): i for i, n in enumerate(node_list)}


def _attach_voltages_to_nodes(sim_data, node_list):
    """
    Fix the original bug: bus names returned by OpenDSS are lowercase,
    but graph node names are mixed case. Use a case-insensitive lookup
    and report match counts so failures are visible.
    """
    if sim_data is None:
        return {"matched": 0, "unmatched": 0}

    lbm = sim_data.get("load_bus_map", {})
    ni_lower = _safe_node_lookup(node_list)
    stats = {"matched": 0, "unmatched": 0, "scenarios": {}}

    for sc in ("baseline", "qp"):
        if sc not in sim_data:
            continue
        sd = sim_data[sc]
        nv = {}
        unmatched_for_sc = 0
        for lname, vs in sd["voltages"].items():
            bus = lbm.get(lname, "").lower()
            idx = ni_lower.get(bus)
            if idx is not None:
                nv[str(idx)] = vs
            else:
                unmatched_for_sc += 1
        sd["nv"] = nv
        stats["scenarios"][sc] = {
            "matched": len(nv),
            "unmatched": unmatched_for_sc,
        }
        stats["matched"] += len(nv)
        stats["unmatched"] += unmatched_for_sc
        logger.info("Mapped %d voltages, %d unmatched for %s",
                    len(nv), unmatched_for_sc, sc)
    return stats


def _serialise_topology(G, pos):
    """Convert NetworkX graph to flat arrays for the frontend."""
    nl = list(G.nodes())
    ni = {n: i for i, n in enumerate(nl)}

    nx_a = [round(float(pos[n][0]), 5) for n in nl]
    ny_a = [round(float(pos[n][1]), 5) for n in nl]
    vl_a = [{"HV": 0, "MV": 1, "LV": 2}.get(G.nodes[n].get("vl", "LV"), 2)
            for n in nl]
    hl_a = [1 if G.nodes[n].get("hl", False) else 0 for n in nl]
    nloads = [G.nodes[n].get("nl", 0) for n in nl]
    dp_a = [G.nodes[n].get("depth", 999) for n in nl]
    dist_a = [round(G.nodes[n].get("dist_m", 0.0), 1) for n in nl]
    tx_a = [G.nodes[n].get("tx", "") for n in nl]

    ea, eb, et = [], [], []
    for u, v, d in G.edges(data=True):
        if u in ni and v in ni:
            du = G.nodes[u].get("depth", 999)
            dv = G.nodes[v].get("depth", 999)
            if du <= dv:
                ea.append(ni[u])
                eb.append(ni[v])
            else:
                ea.append(ni[v])
                eb.append(ni[u])
            e = d.get("element", "line")
            et.append(0 if e in ("transformer", "regulator") else 1)

    return {
        "nl": nl, "ni": ni,
        "nx": nx_a, "ny": ny_a, "vl": vl_a,
        "hl": hl_a, "nloads": nloads, "dp": dp_a,
        "dist": dist_a, "tx": tx_a,
        "ea": ea, "eb": eb, "et": et,
    }


def generate_dashboard(G, source, sim_data=None,
                       output_path="dashboard.html",
                       layout="hierarchical"):
    """Render the full HTML dashboard with all improvements baked in."""
    # Layout
    if layout == "hierarchical":
        logger.info("Computing hierarchical layout ...")
        pos = hierarchical_layout(G, source)
    else:
        logger.info("Computing spring layout (~30s for %d nodes) ...",
                    G.number_of_nodes())
        pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)

    topo = _serialise_topology(G, pos)
    nl = topo["nl"]

    # Attach voltage data to nodes (with the case-fix)
    match_stats = _attach_voltages_to_nodes(sim_data, nl)

    # Build the health summary the frontend will display (#2, #10)
    has_sim = sim_data is not None
    health_payload = {}
    if has_sim:
        h = sim_data.get("health", {})
        health_payload = {
            "monitors_added": h.get("monitors_added", 0),
            "monitors_requested": h.get("monitors_requested", 0),
            "load_bus_map_size": h.get("load_bus_map_size", 0),
            "load_bus_map_failures": h.get("load_bus_map_failures", 0),
            "loads_in_circuit": h.get("loads_in_circuit", 0),
            "tx_monitor_ok": h.get("tx_monitor_ok", False),
            "voltages_matched": match_stats["matched"],
            "voltages_unmatched": match_stats["unmatched"],
            "baseline_converged": (
                sim_data.get("baseline", {}).get("health", {})
                .get("converged")
            ),
            "qp_converged": (
                sim_data.get("qp", {}).get("health", {}).get("converged")
            ),
            "date": sim_data.get("baseline", {}).get("date", ""),
        }

    # Assemble the data payload for embedding in the HTML.
    # NOTE on TX power sign: OpenDSS Monitor mode=1 at terminal=2 reports
    # power flowing INTO that terminal — so when loads consume, the value
    # is negative (power flows OUT of the LV terminal toward the loads).
    # Flip sign once here so that "tp > 0 = import (HV→LV→loads)" is
    # the convention everywhere downstream. This matches how a network
    # operator naturally reads a substation power chart.
    def _flip(arr):
        return [-x for x in arr]

    if has_sim:
        sim_js = json.dumps({
            "baseline": {
                "nv": sim_data["baseline"]["nv"],
                "tp": _flip(sim_data["baseline"]["tx_p"]),
                "lk": sim_data["baseline"]["loss_kw"],
            },
            "qp": {
                "nv": sim_data["qp"]["nv"],
                "tp": _flip(sim_data["qp"]["tx_p"]),
                "lk": sim_data["qp"]["loss_kw"],
            },
            "date": sim_data.get("baseline", {}).get("date", ""),
        })
    else:
        sim_js = "null"

    # Network statistics for the sidebar
    n_mv = sum(1 for v in topo["vl"] if v == 1)
    n_lv = sum(1 for v in topo["vl"] if v == 2)
    n_loads = sum(topo["hl"])
    n_tx = sum(1 for t in topo["et"] if t == 0)
    total_loads = sum(topo["nloads"])

    html = _render_html(
        topo=topo,
        sim_js=sim_js,
        health_payload=health_payload,
        has_sim=has_sim,
        n_mv=n_mv, n_lv=n_lv, n_loads=n_loads,
        n_tx=n_tx, total_loads=total_loads,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(output_path) // 1024
    logger.info("Dashboard written: %s (%d KB)", output_path, size_kb)
    return output_path


# ==================================================================
# HTML / JS RENDERING
# ==================================================================

def _render_html(topo, sim_js, health_payload, has_sim,
                 n_mv, n_lv, n_loads, n_tx, total_loads):
    """
    Render the dashboard HTML. Kept in one place rather than inlined into
    generate_dashboard so the markup is easier to read. The JS payload
    handles all interactive features end to end.
    """
    hrs = [f"{h//2}:{(h%2)*30:02d}" for h in range(48)]
    health_js = json.dumps(health_payload)
    mt_disp = "flex" if has_sim else "none"
    pbx_disp = "flex" if has_sim else "none"

    # Pre-compute a few JSON blobs to keep the f-string readable
    NX_J = json.dumps(topo["nx"])
    NY_J = json.dumps(topo["ny"])
    VL_J = json.dumps(topo["vl"])
    HL_J = json.dumps(topo["hl"])
    NLD_J = json.dumps(topo["nloads"])
    DP_J = json.dumps(topo["dp"])
    DIST_J = json.dumps(topo["dist"])
    TX_J = json.dumps(topo["tx"])
    EA_J = json.dumps(topo["ea"])
    EB_J = json.dumps(topo["eb"])
    ET_J = json.dumps(topo["et"])
    NM_J = json.dumps(topo["nl"])
    HRS_J = json.dumps(hrs)

    # Always render the chart <div>s so initCharts() can populate them
    # later (server mode hot-swap). When there's no data, show a
    # "no data" overlay using CSS until data arrives.
    bottom_panels = (
        "<div class='CP'><div id='pc' class='chart'></div>"
        "<div class='CP-empty' id='pc-empty'>"
        "Run with --simulate or click LOAD"
        "</div></div>"
        "<div class='CP'><div id='hc' class='chart'></div>"
        "<div class='CP-empty' id='hc-empty'>"
        "Voltage distribution (no data yet)"
        "</div></div>"
        "<div class='CP'><div id='vc' class='chart'></div>"
        "<div class='CP-empty' id='vc-empty'>"
        "Violation timeline (no data yet)"
        "</div></div>"
    )

    # CSS — minor tweak: 3 columns on bottom, health panel in sidebar
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Elermore Vale — Power Flow Dashboard v2</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Outfit:wght@300;400;600;700&display=swap');
:root{{--bg:#04070d;--pn:#0a1018;--cd:#101a28;--bd:#162240;--tx:#b8c8da;
  --dm:#4a5e78;--ac:#00d4ff;--aw:#ffd600;--ag:#00e676;--ar:#ff1744;--ao:#ff6b35}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--tx);
  height:100vh;overflow:hidden}}
.D{{display:grid;grid-template-rows:42px minmax(280px,1fr) 160px 160px;
  grid-template-columns:1fr 320px;height:100vh;gap:1px;background:var(--bd)}}
.NP{{background:var(--bg);position:relative;min-height:0;overflow:hidden}}
.H{{grid-column:1/-1;background:var(--pn);display:flex;align-items:center;
  justify-content:space-between;padding:0 16px}}
.HL{{display:flex;align-items:center;gap:12px}}
.H h1{{font-family:'IBM Plex Mono',monospace;font-size:.9rem;font-weight:700;
  color:var(--ac)}}
.VT,.MT{{display:flex;background:var(--bg);border-radius:4px;overflow:hidden;
  border:1px solid var(--bd)}}
.VB,.MB{{padding:4px 12px;font-family:'IBM Plex Mono',monospace;
  font-size:.6rem;font-weight:600;border:none;cursor:pointer;background:0;
  color:var(--dm);transition:.12s}}
.VB.on{{background:#1e3a5f;color:var(--ac)}}
.MB.on{{background:var(--ac);color:var(--bg)}}
.VB:hover:not(.on),.MB:hover:not(.on){{color:var(--tx)}}
.PB{{display:flex;align-items:center;gap:8px}}
.PBT{{width:28px;height:28px;border-radius:50%;border:2px solid var(--ac);
  background:0;color:var(--ac);cursor:pointer;display:flex;
  align-items:center;justify-content:center;font-size:.8rem}}
.PBT:hover{{background:var(--ac);color:var(--bg)}}
.TS{{width:140px;accent-color:var(--ac);cursor:pointer}}
.TD{{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:var(--ac);
  min-width:40px}}
.SBn{{padding:3px 6px;font-family:'IBM Plex Mono',monospace;font-size:.55rem;
  border:1px solid var(--bd);background:0;color:var(--dm);border-radius:3px;
  cursor:pointer}}
.SBn.on{{border-color:var(--ac);color:var(--ac)}}
canvas{{width:100%;height:100%;display:block}}
.SI{{background:var(--pn);padding:12px;overflow-y:auto;display:flex;
  flex-direction:column;gap:10px;font-size:.8rem}}
.SI h3{{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--dm);
  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:2px}}
.SG{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.SC{{background:var(--cd);border:1px solid var(--bd);border-radius:4px;
  padding:7px 9px}}
.SC .L{{font-size:.55rem;color:var(--dm);text-transform:uppercase;
  letter-spacing:.4px}}
.SC .V{{font-family:'IBM Plex Mono',monospace;font-size:.95rem;
  font-weight:700;margin-top:1px}}
.V.w{{color:var(--aw)}}.V.d{{color:var(--ar)}}.V.g{{color:var(--ag)}}
.HP{{display:flex;flex-direction:column;gap:3px}}
.HR{{display:flex;align-items:center;gap:6px;font-size:.7rem;
  font-family:'IBM Plex Mono',monospace}}
.HD{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.HD.ok{{background:var(--ag);box-shadow:0 0 5px var(--ag)}}
.HD.warn{{background:var(--aw);box-shadow:0 0 5px var(--aw)}}
.HD.err{{background:var(--ar);box-shadow:0 0 5px var(--ar)}}
.HD.nul{{background:var(--dm)}}
.CB{{height:10px;border-radius:2px;
  background:linear-gradient(90deg,#1565c0,#42a5f5,#cfd8dc,#ef5350,#b71c1c);
  margin:3px 0 1px}}
.CBL{{display:flex;justify-content:space-between;
  font-family:'IBM Plex Mono',monospace;font-size:.5rem;color:var(--dm)}}
.LR{{display:flex;align-items:center;gap:6px;font-size:.72rem;margin:1px 0}}
.LD{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}
.BP{{grid-column:1/-1;background:var(--pn);display:grid;
  grid-template-columns:1fr 1fr 1fr;gap:1px;min-height:0;overflow:hidden}}
.HM{{grid-column:1/-1;background:var(--pn);display:flex;
  flex-direction:column;position:relative;min-height:0;overflow:hidden}}
.HM-head{{display:flex;align-items:center;gap:8px;padding:4px 12px;
  border-bottom:1px solid var(--bd)}}
.HM-head h4{{font-family:'IBM Plex Mono',monospace;font-size:.65rem;
  color:var(--dm);font-weight:600;text-transform:uppercase;
  letter-spacing:1.1px}}
.HM-tabs{{display:flex;background:var(--bg);border-radius:4px;overflow:hidden;
  border:1px solid var(--bd)}}
.HM-tab{{padding:3px 10px;font-family:'IBM Plex Mono',monospace;
  font-size:.55rem;font-weight:600;border:none;cursor:pointer;
  background:0;color:var(--dm);transition:.12s}}
.HM-tab.on{{background:var(--ac);color:var(--bg)}}
.HM-tab:hover:not(.on){{color:var(--tx)}}
#hm{{flex:1;width:100%;min-height:0}}
.CP{{background:var(--bg);padding:2px;position:relative}}
#pc,#hc,#vc{{width:100%;height:100%}}
.CP-empty{{position:absolute;inset:0;display:flex;
  align-items:center;justify-content:center;font-size:.75rem;
  color:var(--dm);text-align:center;padding:12px;pointer-events:none;
  font-family:'IBM Plex Mono',monospace}}
.CP-empty.hidden{{display:none}}
.NS{{display:flex;align-items:center;justify-content:center;height:100%;
  font-size:.75rem;color:var(--dm);text-align:center;padding:12px}}
.TT{{position:absolute;background:var(--cd);border:1px solid var(--ac);
  border-radius:4px;padding:6px 10px;font-size:.72rem;pointer-events:none;
  display:none;z-index:10;font-family:'IBM Plex Mono',monospace;
  max-width:300px;line-height:1.4}}
.note{{font-size:.65rem;color:#3a4a5e;line-height:1.4;padding:8px;
  background:var(--bg);border-radius:4px;border:1px solid var(--bd)}}
.note b{{color:var(--dm)}}
.EXP{{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px}}
.EX{{padding:3px 8px;font-family:'IBM Plex Mono',monospace;font-size:.55rem;
  border:1px solid var(--bd);background:0;color:var(--ac);border-radius:3px;
  cursor:pointer}}
.EX:hover{{background:var(--ac);color:var(--bg)}}
</style></head><body>
<div class="D">
<div class="H">
<div class="HL"><h1>ELERMORE VALE v2</h1>
<div class="VT">
<button class="VB on" data-v="static" onclick="setView('static')">STATIC TOPOLOGY</button>
<button class="VB" data-v="dynamic" onclick="setView('dynamic')">LIVE FLOW</button>
</div>
<div class="MT" id="mt" style="display:{mt_disp}">
<button class="MB on" data-m="baseline" onclick="setMode('baseline')">BASELINE</button>
<button class="MB" data-m="qp" onclick="setMode('qp')">QP DISPATCH</button>
</div></div>
<div class="PB" id="pbx" style="display:{pbx_disp}">
<button class="PBT" id="pb" onclick="togglePlay()">&#9654;</button>
<input type="range" class="TS" id="ts" min="0" max="47" value="0" oninput="setTime(+this.value)">
<span class="TD" id="td">0:00</span>
<button class="SBn on" data-s="1" onclick="setSpd(1)">1x</button>
<button class="SBn" data-s="2" onclick="setSpd(2)">2x</button>
<button class="SBn" data-s="4" onclick="setSpd(4)">4x</button>
</div></div>

<div class="NP"><canvas id="cv"></canvas><div class="TT" id="tt"></div></div>

<div class="SI">

<h3>Data Health</h3>
<div class="HP" id="health"></div>

<h3>Live Statistics</h3>
<div class="SG">
<div class="SC"><div class="L">V Min</div><div class="V" id="sn">&mdash;</div></div>
<div class="SC"><div class="L">V Max</div><div class="V" id="sx">&mdash;</div></div>
<div class="SC"><div class="L">Violations</div><div class="V" id="sv">&mdash;</div></div>
<div class="SC"><div class="L">TX Power</div><div class="V" id="sp">&mdash;</div></div>
</div>

<h3>Voltage Scale</h3><div class="CB"></div>
<div class="CBL"><span>0.90</span><span>0.94</span><span>1.00</span>
<span>1.06</span><span>1.10</span></div>

<h3>Topology Legend</h3>
<div class="LR"><div class="LD" style="background:#e74c3c"></div>132 kV Source</div>
<div class="LR"><div class="LD" style="background:#f39c12"></div>11 kV MV ({n_mv})</div>
<div class="LR"><div class="LD" style="background:#2ecc71"></div>LV Load ({n_loads})</div>
<div class="LR"><div class="LD" style="background:#7f8fa6"></div>LV Junction ({n_lv - n_loads})</div>
<div class="LR" style="margin-top:4px"><div class="LD"
  style="background:var(--aw);box-shadow:0 0 6px var(--aw)"></div>
  Yellow = import (sub→loads)</div>
<div class="LR"><div class="LD"
  style="background:var(--ac);box-shadow:0 0 6px var(--ac)"></div>
  Cyan = export (loads→grid)</div>

<h3>Network</h3>
<div class="SG">
<div class="SC"><div class="L">132/11 kV Sub</div><div class="V">50 MVA</div></div>
<div class="SC"><div class="L">MV Feeder</div><div class="V">{n_mv} buses</div></div>
<div class="SC"><div class="L">Dist. TXs</div><div class="V">{n_tx}</div></div>
<div class="SC"><div class="L">Total Buses</div><div class="V">{len(topo['nl'])}</div></div>
<div class="SC"><div class="L">Branches</div><div class="V">{len(topo['ea'])}</div></div>
<div class="SC"><div class="L">Total Loads</div><div class="V">{total_loads}</div></div>
</div>

<h3>Voltage Limits (AS 60038)</h3>
<div class="SC"><div class="L">Nominal</div><div class="V">230 V (1.00 p.u.)</div></div>
<div class="SG" style="margin-top:4px">
<div class="SC"><div class="L">Upper</div><div class="V">253 V (+10%)</div></div>
<div class="SC"><div class="L">Lower</div><div class="V">216 V (&#8722;6%)</div></div>
</div>

<h3>Export</h3>
<div class="EXP">
<button class="EX" onclick="exportPNG()">View → PNG</button>
<button class="EX" onclick="exportCSV()">Timestep → CSV</button>
<button class="EX" onclick="copyURL()">Copy share URL</button>
</div>

<div class="note"><b>About:</b> Elermore Vale, Newcastle NSW &mdash;
~1,785 residential customers, 23 distribution transformers,
31.68 km radial 11 kV feeder. Hierarchical layout — depth runs
left-to-right from the substation. Scroll to zoom, drag to pan.
URL params: <code>?day=190&amp;hour=18&amp;scenario=qp&amp;view=dynamic</code></div>

</div>

<div class="HM">
<div class="HM-head">
<h4>Voltage heatmap — bus × time</h4>
<div class="HM-tabs">
<button class="HM-tab on" data-h="baseline" onclick="setHeatmap('baseline')">BASELINE</button>
<button class="HM-tab" data-h="qp" onclick="setHeatmap('qp')">QP</button>
<button class="HM-tab" data-h="delta" onclick="setHeatmap('delta')">&Delta; (BASE&minus;QP)</button>
</div>
<span id="hm-empty" class="CP-empty"
  style="position:relative;flex:1;text-align:right;padding-right:12px">
  Heatmap will appear once data loads</span>
</div>
<div id="hm"></div>
</div>

<div class="BP">{bottom_panels}</div>

</div>

<script>
// Topology
const NX={NX_J},NY={NY_J},VL={VL_J},HL={HL_J},NLd={NLD_J},
DP={DP_J},DIST={DIST_J},TXNAME={TX_J},
EA={EA_J},EB={EB_J},ET={ET_J},NM={NM_J},hrs={HRS_J};
let hasSim={'true' if has_sim else 'false'};
let S={sim_js};
const HEALTH={health_js};

// =================== URL state (#7) ===================
const URLp=new URLSearchParams(window.location.search);
function urlGet(k,d){{return URLp.has(k)?URLp.get(k):d}}
function urlSet(k,v){{URLp.set(k,v);
  history.replaceState(null,'','?'+URLp.toString())}}

// =================== Health badges (#2,#10) ===================
function renderHealth(){{
  const el=document.getElementById('health');
  if(!hasSim){{el.innerHTML='<div class="HR"><div class="HD nul"></div>'+
    'Topology only — no simulation data</div>';return}}
  const rows=[];
  const tot=HEALTH.monitors_added;
  const req=HEALTH.monitors_requested;
  rows.push(rowFor(tot===req?'ok':(tot>0?'warn':'err'),
    'Voltage monitors',tot+'/'+req));
  const vm=HEALTH.voltages_matched,vu=HEALTH.voltages_unmatched;
  rows.push(rowFor(vu===0?'ok':(vm>0?'warn':'err'),
    'Bus mapping',vm+' matched, '+vu+' lost'));
  rows.push(rowFor(HEALTH.tx_monitor_ok?'ok':'err',
    'TX power monitor',HEALTH.tx_monitor_ok?'recording':'no data'));
  const lf=HEALTH.load_bus_map_failures;
  rows.push(rowFor(lf===0?'ok':'warn',
    'Load→bus lookup',lf===0?'OK':lf+' failed'));
  function st(c){{
    if(c===true)return['ok','converged'];
    if(c==='max_iter_warning')return['warn','max-iter (#485)'];
    if(c===false)return['err','failed'];
    return['nul','unknown'];}}
  const bs=st(HEALTH.baseline_converged);
  rows.push(rowFor(bs[0],'Baseline solver',bs[1]));
  const qs=st(HEALTH.qp_converged);
  rows.push(rowFor(qs[0],'QP solver',qs[1]));
  el.innerHTML=rows.join('');
}}
function rowFor(level,label,val){{
  return '<div class="HR"><div class="HD '+level+'"></div>'+
    '<span>'+label+': <b>'+val+'</b></span></div>';
}}

// =================== Canvas (network view) ===================
const cv=document.getElementById('cv'),ctx=cv.getContext('2d'),
  tt=document.getElementById('tt');
let W,H,camX=0,camY=0,zoom=1,drag=false,dx0,dy0,hoverNode=-1;
let mode=urlGet('scenario','baseline'),
    ct=parseInt(urlGet('hour','0')),
    playing=false,spd=1,lt=0,gt=0,
    view=urlGet('view','static');
function resize(){{
  W=cv.parentElement.clientWidth;H=cv.parentElement.clientHeight;
  cv.width=W*devicePixelRatio;cv.height=H*devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}}
window.addEventListener('resize',resize);resize();

function w2s(wx,wy){{return[(wx-camX)*zoom+W/2,(wy-camY)*zoom+H/2]}}
function s2w(sx,sy){{return[(sx-W/2)/zoom+camX,(sy-H/2)/zoom+camY]}}

cv.addEventListener('wheel',e=>{{
  e.preventDefault();
  const[wx,wy]=s2w(e.offsetX,e.offsetY);
  zoom*=(e.deltaY<0?1.15:1/1.15);
  camX=wx-(e.offsetX-W/2)/zoom;camY=wy-(e.offsetY-H/2)/zoom;
  urlSet('zoom',zoom.toFixed(2));
}});
cv.addEventListener('mousedown',e=>{{
  drag=true;dx0=e.offsetX;dy0=e.offsetY;cv.style.cursor='grabbing';
}});
cv.addEventListener('mousemove',e=>{{
  if(drag){{
    camX-=(e.offsetX-dx0)/zoom;camY-=(e.offsetY-dy0)/zoom;
    dx0=e.offsetX;dy0=e.offsetY;
  }}else{{
    const[wx,wy]=s2w(e.offsetX,e.offsetY);
    let best=-1,bd=1e9;
    for(let i=0;i<NX.length;i++){{
      const d=(NX[i]-wx)**2+(NY[i]-wy)**2;
      if(d<bd){{bd=d;best=i}}
    }}
    const thr=(15/zoom)**2;
    hoverNode=bd<thr?best:-1;
    if(hoverNode>=0){{
      tt.style.display='block';
      tt.style.left=(e.offsetX+14)+'px';
      tt.style.top=(e.offsetY-10)+'px';
      let h='<b>'+NM[hoverNode]+'</b>'+
        '<br>'+['132 kV','11 kV','LV'][VL[hoverNode]];
      if(NLd[hoverNode]>0)h+='<br>Loads: '+NLd[hoverNode];
      h+='<br>Depth: '+DP[hoverNode]+'  /  '+
        (DIST[hoverNode]/1000).toFixed(2)+' km from src';
      if(TXNAME[hoverNode])h+='<br>TX: '+TXNAME[hoverNode];
      if(hasSim&&view==='dynamic'){{
        const nvB=S.baseline.nv[hoverNode];
        const nvQ=S.qp.nv[hoverNode];
        if(nvB&&nvQ){{
          const dv=nvB[ct]-nvQ[ct];
          h+='<br>V (baseline): <b>'+nvB[ct].toFixed(3)+'</b>';
          h+='<br>V (QP): <b>'+nvQ[ct].toFixed(3)+'</b>';
          h+='<br>Δ: <b>'+(dv*1000).toFixed(0)+' mV</b>';
          let viols=0;
          for(let k=0;k<48;k++){{
            const v=S[mode].nv[hoverNode][k];
            if(v<.94||v>1.10)viols++;
          }}
          h+='<br>Violations today ('+mode+'): '+viols+'/48';
        }}
      }}
      tt.innerHTML=h;
    }}else tt.style.display='none';
  }}
}});
cv.addEventListener('mouseup',()=>{{drag=false;cv.style.cursor='default'}});
cv.addEventListener('mouseleave',()=>{{
  drag=false;tt.style.display='none';hoverNode=-1;
}});

const DC=VL.map((v,i)=>v===0?'#e74c3c':v===1?'#f39c12':
  HL[i]?'#2ecc71':'#7f8fa6');
const DS=VL.map((v,i)=>v===0?8:v===1?5:HL[i]?2.8:1.8);

function v2c(v){{
  if(v<=0)return'#222';
  const t=Math.max(0,Math.min(1,(v-.90)/.20));
  if(t<.5){{
    const u=t/.5;
    return`rgb(${{21+203*u|0}},${{101+123*u|0}},${{192+32*u|0}})`;
  }}else{{
    const u=(t-.5)/.5;
    return`rgb(${{224-41*u|0}},${{224-196*u|0}},${{224-196*u|0}})`;
  }}
}}

const NPART=500;
let parts=[];
for(let i=0;i<NPART;i++)parts.push({{
  ei:Math.random()*EA.length|0,
  t:Math.random(),
  sp:.002+Math.random()*.008,
}});

function render(){{
  ctx.clearRect(0,0,W,H);
  ctx.lineWidth=Math.max(0.8,zoom/300);
  ctx.strokeStyle=view==='static'?'#2a4570':'#1e3858';
  ctx.beginPath();
  for(let i=0;i<EA.length;i++)if(ET[i]===1){{
    const[x1,y1]=w2s(NX[EA[i]],NY[EA[i]]);
    const[x2,y2]=w2s(NX[EB[i]],NY[EB[i]]);
    ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
  }}
  ctx.stroke();
  ctx.lineWidth=Math.max(2,zoom/120);
  ctx.strokeStyle=view==='static'?'#4080c0':'#2a5a90';
  ctx.beginPath();
  for(let i=0;i<EA.length;i++)if(ET[i]===0){{
    const[x1,y1]=w2s(NX[EA[i]],NY[EA[i]]);
    const[x2,y2]=w2s(NX[EB[i]],NY[EB[i]]);
    ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
  }}
  ctx.stroke();
  if(view==='dynamic'){{
    // After Python flips the OpenDSS sign so positive=import:
    //   tp[ct] >= 0  →  normal flow (HV→LV→loads), particles run yellow
    //   tp[ct] <  0  →  reverse flow (export to grid), particles run cyan
    const reversed=hasSim&&S&&S[mode]&&S[mode].tp[ct]<0;
    const pr=Math.max(1.5,2.2*zoom/400);
    if(reversed){{
      ctx.shadowColor='#00d4ff';ctx.shadowBlur=3;
      ctx.fillStyle='rgba(0,212,255,0.9)';
    }}else{{
      ctx.shadowColor='#ffd600';ctx.shadowBlur=3;
      ctx.fillStyle='rgba(255,214,0,0.9)';
    }}
    for(let p of parts){{
      const a=EA[p.ei],b=EB[p.ei];
      const u=reversed?(1-p.t):p.t;
      const px=NX[a]+(NX[b]-NX[a])*u;
      const py=NY[a]+(NY[b]-NY[a])*u;
      const[sx,sy]=w2s(px,py);
      ctx.beginPath();ctx.arc(sx,sy,pr,0,6.28);ctx.fill();
    }}
    ctx.shadowBlur=0;
  }}
  const isDyn=view==='dynamic'&&hasSim;
  const nc=isDyn?NX.map((_,i)=>{{
    const nv=S[mode].nv[i];return nv?v2c(nv[ct]):DC[i];
  }}):DC;
  for(let i=0;i<NX.length;i++){{
    const[sx,sy]=w2s(NX[i],NY[i]);
    const r=DS[i]*Math.max(.5,zoom/400);
    if(isDyn){{
      const nv=S[mode].nv[i];
      if(nv){{
        const v=nv[ct];
        if(v>0&&(v<.94||v>1.10)){{
          const pulse=.3+.25*Math.sin(gt*5);
          ctx.fillStyle=v<.94?
            `rgba(21,101,192,${{pulse}})`:
            `rgba(183,28,28,${{pulse}})`;
          ctx.beginPath();ctx.arc(sx,sy,r*5,0,6.28);ctx.fill();
        }}
      }}
    }}
    ctx.fillStyle=nc[i];
    ctx.beginPath();ctx.arc(sx,sy,r,0,6.28);ctx.fill();
  }}
  if(hoverNode>=0){{
    const[sx,sy]=w2s(NX[hoverNode],NY[hoverNode]);
    ctx.strokeStyle='#00d4ff';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(sx,sy,10,0,6.28);ctx.stroke();
  }}
}}
let lf=performance.now();
function loop(now){{
  const dt=(now-lf)/1000;lf=now;gt+=dt;
  if(view==='dynamic')for(let p of parts){{
    p.t+=p.sp*dt*60;
    if(p.t>=1){{
      p.t=0;
      p.ei=Math.random()*EA.length|0;
      p.sp=.002+Math.random()*.008;
    }}
  }}
  render();
  if(playing&&view==='dynamic'&&now-lt>=500/spd){{
    lt=now;ct=(ct+1)%48;setTime(ct);
  }}
  requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);

// =================== View / mode / time ===================
function setView(v){{
  view=v;urlSet('view',v);
  document.querySelectorAll('.VB').forEach(b=>
    b.classList.toggle('on',b.dataset.v===v));
  const pbx=document.getElementById('pbx');
  if(pbx)pbx.style.display=(v==='dynamic'&&hasSim)?'flex':'none';
  const mt=document.getElementById('mt');
  if(mt)mt.style.display=(v==='dynamic'&&hasSim)?'flex':'none';
  if(v==='static'){{
    playing=false;
    document.getElementById('pb').innerHTML='&#9654;';
  }}
}}

function setTime(t){{
  ct=t;urlSet('hour',t);
  document.getElementById('ts').value=t;
  document.getElementById('td').textContent=hrs[t];
  if(!hasSim)return;
  const sc=S[mode],nv=sc.nv;
  const aV=Object.values(nv).map(a=>a[t]).filter(v=>v>.01);
  if(typeof Plotly!=='undefined'){{
    if(document.getElementById('hc')){{
      try{{Plotly.restyle('hc',{{x:[aV]}},[0])}}catch(e){{}}
    }}
    // sync time cursor on all four charts (#3)
    const hr=t*.5;
    if(document.getElementById('pc')){{
      try{{Plotly.restyle('pc',{{x:[[hr,hr]]}},[2])}}catch(e){{}}
    }}
    if(document.getElementById('vc')){{
      try{{Plotly.restyle('vc',{{x:[[hr,hr]]}},[4])}}catch(e){{}}
    }}
    if(document.getElementById('hm')){{
      try{{Plotly.restyle('hm',{{x:[[hr,hr]]}},[1])}}catch(e){{}}
    }}
  }}
  if(view!=='dynamic')return;
  if(aV.length){{
    const mn=Math.min(...aV),mx=Math.max(...aV);
    const vi=aV.filter(v=>v<.94||v>1.10).length;
    document.getElementById('sn').textContent=mn.toFixed(3);
    document.getElementById('sx').textContent=mx.toFixed(3);
    document.getElementById('sv').textContent=vi;
    document.getElementById('sn').className='V'+
      (mn<.94?' d':mn<.97?' w':' g');
    document.getElementById('sx').className='V'+
      (mx>1.10?' d':mx>1.07?' w':' g');
    document.getElementById('sv').className='V'+(vi>0?' d':' g');
  }}
  const p=sc.tp[t];
  if(p!==undefined)document.getElementById('sp').textContent=
    Math.round(p)+' kW';
}}

function setMode(m){{
  mode=m;urlSet('scenario',m);
  document.querySelectorAll('.MB').forEach(b=>
    b.classList.toggle('on',b.dataset.m===m));
  if(hasSim)setTime(ct);
}}
function togglePlay(){{
  playing=!playing;
  document.getElementById('pb').innerHTML=playing?'&#9646;&#9646;':'&#9654;';
  if(playing)lt=performance.now();
}}
function setSpd(s){{
  spd=s;
  document.querySelectorAll('.SBn').forEach(b=>
    b.classList.toggle('on',+b.dataset.s===s));
}}

// =================== Charts (#3, #4) ===================
function computeViolationCounts(){{
  // Returns {{base_under,base_over,qp_under,qp_over}} as 48-arrays.
  const out={{bu:Array(48).fill(0),bo:Array(48).fill(0),
             qu:Array(48).fill(0),qo:Array(48).fill(0)}};
  if(!S)return out;
  for(const sc of ['baseline','qp']){{
    const nv=S[sc].nv;
    for(const k in nv){{
      const arr=nv[k];
      for(let t=0;t<48;t++){{
        const v=arr[t];
        if(v<=0)continue;
        if(v<.94){{
          if(sc==='baseline')out.bu[t]++;else out.qu[t]++;
        }}else if(v>1.10){{
          if(sc==='baseline')out.bo[t]++;else out.qo[t]++;
        }}
      }}
    }}
  }}
  return out;
}}

// =================== Heatmap (Plotly) ===================
// Cache the sorted bus-index list so we don't re-sort on every redraw.
let hmOrder=null;
let hmMode='baseline';

function buildHmOrder(){{
  // Sort monitored buses by BFS depth (substation→leaves).
  // Buses near the source plot at the top, far feeders at the bottom —
  // typically that's where the worst violations live, so they stand out.
  if(!S||!S.baseline)return[];
  const ids=Object.keys(S.baseline.nv).map(Number);
  ids.sort((a,b)=>(DP[a]||0)-(DP[b]||0)||a-b);
  return ids;
}}

function buildHmMatrix(scenario){{
  // Returns 2D array shape [n_buses, 48] for Plotly heatmap.
  // For 'delta' returns baseline − qp (positive = QP improved overvoltage).
  if(!S||!hmOrder)return{{z:[],ids:[]}};
  const z=[];
  if(scenario==='delta'){{
    for(const i of hmOrder){{
      const b=S.baseline.nv[i],q=S.qp.nv[i];
      if(!b||!q){{z.push(new Array(48).fill(null));continue}}
      const row=new Array(48);
      for(let t=0;t<48;t++)row[t]=(b[t]-q[t]);
      z.push(row);
    }}
  }}else{{
    for(const i of hmOrder){{
      const v=S[scenario].nv[i];
      z.push(v?v.slice():new Array(48).fill(null));
    }}
  }}
  return{{z:z,ids:hmOrder}};
}}

function drawHeatmap(){{
  if(!hasSim||!S)return;
  if(!hmOrder)hmOrder=buildHmOrder();
  const{{z,ids}}=buildHmMatrix(hmMode);
  const xH=hrs.map((_,i)=>i*.5);
  // y axis = bus rank (1..N) — actual names live in hover via customdata
  const y=ids.map((_,i)=>i+1);
  const customdata=ids.map(i=>NM[i]);

  let colorscale,zmin,zmax,zmid,ttl;
  if(hmMode==='delta'){{
    // Diverging: blue (QP higher) → white (no change) → red (QP lower)
    let absmax=0;
    for(const row of z)for(const v of row)
      if(v!=null&&Math.abs(v)>absmax)absmax=Math.abs(v);
    absmax=Math.max(absmax,0.001);
    zmin=-absmax;zmax=absmax;zmid=0;
    colorscale=[[0,'#1565c0'],[.5,'#cfd8dc'],[1,'#b71c1c']];
    ttl='Δ voltage (p.u.) — base − QP';
  }}else{{
    zmin=0.92;zmax=1.12;zmid=1.0;
    colorscale=[[0,'#1565c0'],[.1,'#42a5f5'],[.4,'#cfd8dc'],
                [.6,'#cfd8dc'],[.9,'#ef5350'],[1,'#b71c1c']];
    ttl='Voltage (p.u.) — '+hmMode;
  }}

  Plotly.newPlot('hm',[{{
    z:z,x:xH,y:y,customdata:customdata,
    type:'heatmap',colorscale:colorscale,
    zmin:zmin,zmax:zmax,zmid:zmid,
    hovertemplate:'bus %{{customdata}}<br>'+
      'hour %{{x}}<br>'+
      (hmMode==='delta'?'Δ ':'V ')+'%{{z:.4f}} p.u.<extra></extra>',
    colorbar:{{title:{{text:ttl,font:{{size:9}}}},
      tickfont:{{size:8}},len:0.9,thickness:10}},
  }},{{
    // overlay: vertical "now" cursor
    x:[0,0],y:[0,Math.max(1,y.length)],mode:'lines',
    line:{{color:'#ffd600',width:1.5,dash:'dot'}},
    showlegend:false,name:'cur',hoverinfo:'skip',
  }}],{{
    paper_bgcolor:'#04070d',plot_bgcolor:'#0a1018',
    font:{{family:'IBM Plex Mono',color:'#4a5e78',size:9}},
    xaxis:{{title:'Hour',gridcolor:'#162240',range:[0,24],
      side:'bottom'}},
    yaxis:{{title:'Bus rank (depth-sorted)',gridcolor:'#162240',
      autorange:'reversed'}},
    margin:{{t:6,b:32,l:60,r:6}}
  }},{{displayModeBar:false,responsive:true}});

  const empty=document.getElementById('hm-empty');
  if(empty)empty.style.display='none';
}}

function setHeatmap(m){{
  hmMode=m;
  document.querySelectorAll('.HM-tab').forEach(b=>
    b.classList.toggle('on',b.dataset.h===m));
  drawHeatmap();
}}

// Wrapped so server mode can re-init when day changes
let chartsInit=false;
function initCharts(){{
  if(!hasSim||!S)return;
  const bP=S.baseline.tp,qP=S.qp.tp,xH=hrs.map((_,i)=>i*.5);
  const yMin=Math.min(...bP,...qP)*1.1;
  const yMax=Math.max(...bP,...qP)*1.1;
  Plotly.newPlot('pc',[
    {{x:xH,y:bP,name:'Baseline',
      line:{{color:'#ff6b35',width:2}},type:'scatter'}},
    {{x:xH,y:qP,name:'QP',
      line:{{color:'#00d4ff',width:2}},type:'scatter'}},
    {{x:[0,0],y:[yMin,yMax],mode:'lines',
      line:{{color:'#ffd600',width:1.5,dash:'dot'}},
      showlegend:false,name:'cur'}}
  ],{{
    paper_bgcolor:'#04070d',plot_bgcolor:'#0a1018',
    font:{{family:'IBM Plex Mono',color:'#4a5e78',size:9}},
    xaxis:{{title:'Hour',gridcolor:'#162240',range:[0,24]}},
    yaxis:{{title:'TX Power (kW)',gridcolor:'#162240'}},
    legend:{{x:.02,y:.98,bgcolor:'rgba(0,0,0,0)'}},
    margin:{{t:6,b:32,l:48,r:6}}
  }},{{displayModeBar:false,responsive:true}});

  Plotly.newPlot('hc',[
    {{x:[],type:'histogram',marker:{{color:'#00d4ff'}},nbinsx:30}}
  ],{{
    paper_bgcolor:'#04070d',plot_bgcolor:'#0a1018',
    font:{{family:'IBM Plex Mono',color:'#4a5e78',size:9}},
    xaxis:{{title:'V (p.u.)',gridcolor:'#162240',range:[.85,1.15]}},
    yaxis:{{title:'Count',gridcolor:'#162240'}},
    margin:{{t:6,b:32,l:36,r:6}},
    shapes:[
      {{type:'line',x0:.94,x1:.94,y0:0,y1:1,yref:'paper',
        line:{{color:'#ff1744',dash:'dash',width:1}}}},
      {{type:'line',x0:1.10,x1:1.10,y0:0,y1:1,yref:'paper',
        line:{{color:'#ff1744',dash:'dash',width:1}}}},
      {{type:'line',x0:1,x1:1,y0:0,y1:1,yref:'paper',
        line:{{color:'#ffffff22',dash:'dot',width:1}}}}
    ]
  }},{{displayModeBar:false,responsive:true}});

  const vc=computeViolationCounts();
  const vmax=Math.max(1,...vc.bu,...vc.bo,...vc.qu,...vc.qo);
  Plotly.newPlot('vc',[
    {{x:xH,y:vc.bu,name:'Base under',type:'bar',
      marker:{{color:'#1565c0'}}}},
    {{x:xH,y:vc.bo,name:'Base over',type:'bar',
      marker:{{color:'#b71c1c'}}}},
    {{x:xH,y:vc.qu,name:'QP under',type:'bar',
      marker:{{color:'#42a5f5'}}}},
    {{x:xH,y:vc.qo,name:'QP over',type:'bar',
      marker:{{color:'#ef5350'}}}},
    {{x:[0,0],y:[0,vmax*1.1],mode:'lines',
      line:{{color:'#ffd600',width:1.5,dash:'dot'}},
      showlegend:false,name:'cur'}}
  ],{{
    barmode:'group',
    paper_bgcolor:'#04070d',plot_bgcolor:'#0a1018',
    font:{{family:'IBM Plex Mono',color:'#4a5e78',size:9}},
    xaxis:{{title:'Hour',gridcolor:'#162240',range:[0,24]}},
    yaxis:{{title:'Violations',gridcolor:'#162240'}},
    legend:{{x:.02,y:.98,bgcolor:'rgba(0,0,0,0)',font:{{size:8}}}},
    margin:{{t:6,b:32,l:36,r:6}}
  }},{{displayModeBar:false,responsive:true}});
  chartsInit=true;
  // Hide the "no data yet" overlays now that charts are populated
  ['pc-empty','hc-empty','vc-empty'].forEach(id=>{{
    const el=document.getElementById(id);
    if(el)el.classList.add('hidden');
  }});
  // Heatmap: rebuild bus order (depends on which buses are monitored)
  hmOrder=null;
  drawHeatmap();
}}
initCharts();

// Hot-swap entry point used by server mode after a /api/sim fetch.
window.applySimData=function(data){{
  if(!data)return;
  // Replace S; first call may be from null (server mode initial load)
  S={{baseline:data.baseline,qp:data.qp}};
  hasSim=true;
  // Show controls hidden in topology-only mode
  const mt=document.getElementById('mt');
  const pbx=document.getElementById('pbx');
  if(mt)mt.style.display='flex';
  if(pbx)pbx.style.display='flex';
  // Re-render charts with new data
  initCharts();
  // Refresh health badges
  if(data.health){{
    Object.assign(HEALTH,data.health);
    HEALTH.date=data.date||HEALTH.date;
    renderHealth();
  }}
  // Reapply current timestep so all UI updates
  setTime(ct);
}};

// =================== Export (#9) ===================
function exportPNG(){{
  const link=document.createElement('a');
  link.download='elermorevale_'+(HEALTH.date||'view')+
    '_'+mode+'_t'+ct+'.png';
  link.href=cv.toDataURL('image/png');
  link.click();
}}
function exportCSV(){{
  // Emit one row per node for the current timestep + scenario.
  if(!hasSim){{alert('Nothing to export — load simulation data');return}}
  const nv=S[mode].nv;
  const lines=['node,bus_name,depth,dist_m,tx,voltage_pu'];
  for(const k in nv){{
    const i=+k;
    lines.push([i,NM[i],DP[i],DIST[i],TXNAME[i]||'',
      nv[k][ct].toFixed(4)].join(','));
  }}
  const blob=new Blob([lines.join('\\n')],{{type:'text/csv'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url;
  a.download='elermorevale_'+mode+'_t'+ct+'.csv';
  a.click();
  URL.revokeObjectURL(url);
}}
function copyURL(){{
  navigator.clipboard.writeText(window.location.href);
  const el=document.createElement('div');
  el.textContent='URL copied';
  el.style.cssText='position:fixed;top:60px;right:20px;'+
    'background:#00d4ff;color:#04070d;padding:8px 14px;'+
    'border-radius:4px;font-family:IBM Plex Mono;font-size:.7rem;'+
    'z-index:1000;font-weight:700';
  document.body.appendChild(el);
  setTimeout(()=>el.remove(),1200);
}}

// =================== Initialise ===================
const mnx=Math.min(...NX),mxx=Math.max(...NX);
const mny=Math.min(...NY),mxy=Math.max(...NY);
camX=(mnx+mxx)/2;camY=(mny+mxy)/2;
const baseZoom=Math.min(W/(mxx-mnx+.01),H/(mxy-mny+.01))*.88;
zoom=parseFloat(urlGet('zoom',baseZoom))||baseZoom;
renderHealth();
setView(view);
if(hasSim)setTime(ct);
setMode(mode);
</script></body></html>"""


# ==================================================================
# OPTIONAL FLASK BACKEND (#1)
# ==================================================================

def run_server(glm_dir, common_dir, profiles_csv, port=8765,
               layout="hierarchical"):
    """
    Start a small Flask server that exposes the topology once and runs
    simulations on demand. The page can switch days/scenarios without
    rebuilding HTML.
    """
    try:
        from flask import Flask, jsonify, request, send_from_directory
    except ImportError:
        logger.error(
            "Flask not installed. pip install flask, "
            "or skip --serve to use static mode."
        )
        return

    G, source = build_topology(glm_dir)
    if layout == "hierarchical":
        pos = hierarchical_layout(G, source)
    else:
        pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)
    topo = _serialise_topology(G, pos)

    app = Flask(__name__)

    # In-memory cache so the same (day) request doesn't re-solve.
    cache = {}

    @app.route("/")
    def index():
        return _server_index_html(topo)

    @app.route("/api/topology")
    def api_topo():
        return jsonify({k: v for k, v in topo.items() if k != "ni"})

    @app.route("/api/sim")
    def api_sim():
        day = int(request.args.get("day", 190))
        if day in cache:
            return jsonify(cache[day])
        try:
            sim = run_simulation(glm_dir, common_dir, profiles_csv, day)
        except Exception as exc:
            logger.exception("Simulation failed")
            return jsonify({"error": str(exc)}), 500
        match_stats = _attach_voltages_to_nodes(sim, topo["nl"])
        sim.setdefault("health", {})["match_stats"] = match_stats
        # strip the heavy load_bus_map; rename tx_p→tp and flip sign so
        # positive = import (see comment in generate_dashboard).
        out = {
            "baseline": {
                "nv": sim["baseline"]["nv"],
                "tp": [-x for x in sim["baseline"]["tx_p"]],
                "lk": sim["baseline"]["loss_kw"],
            },
            "qp": {
                "nv": sim["qp"]["nv"],
                "tp": [-x for x in sim["qp"]["tx_p"]],
                "lk": sim["qp"]["loss_kw"],
            },
            "date": sim["baseline"].get("date", ""),
            "health": {
                "monitors_added": sim["health"].get("monitors_added", 0),
                "monitors_requested": sim["health"].get(
                    "monitors_requested", 0),
                "load_bus_map_size": sim["health"].get(
                    "load_bus_map_size", 0),
                "load_bus_map_failures": sim["health"].get(
                    "load_bus_map_failures", 0),
                "tx_monitor_ok": sim["health"].get("tx_monitor_ok", False),
                "voltages_matched": match_stats["matched"],
                "voltages_unmatched": match_stats["unmatched"],
                "baseline_converged": (
                    sim["baseline"]["health"]["converged"]),
                "qp_converged": sim["qp"]["health"]["converged"],
                "date": sim["baseline"].get("date", ""),
            },
        }
        cache[day] = out
        return jsonify(out)

    logger.info("Serving on http://localhost:%d  (Ctrl-C to stop)", port)
    app.run(host="0.0.0.0", port=port, debug=False)


def _server_index_html(topo):
    """
    Backend-aware HTML page. Renders the same dashboard template with
    sim_js=null (topology-only initial state), then injects a tiny
    day-picker that fetches /api/sim?day=N and calls window.applySimData.
    """
    # Compute the same network stats the static path uses
    n_mv = sum(1 for v in topo["vl"] if v == 1)
    n_lv = sum(1 for v in topo["vl"] if v == 2)
    n_loads = sum(topo["hl"])
    n_tx = sum(1 for t in topo["et"] if t == 0)
    total_loads = sum(topo["nloads"])

    base = _render_html(
        topo=topo,
        sim_js="null",
        health_payload={},
        has_sim=False,
        n_mv=n_mv, n_lv=n_lv, n_loads=n_loads,
        n_tx=n_tx, total_loads=total_loads,
    )

    # Inject day-picker UI + fetch logic just before </body>.
    # We force-show the MT/PB controls because hasSim flips on after fetch.
    picker = """
<div id="dpx" style="position:fixed;top:8px;right:340px;z-index:100;
  display:flex;align-items:center;gap:6px;background:#0a1018;
  padding:6px 10px;border:1px solid #162240;border-radius:4px">
<span style="font-family:'IBM Plex Mono',monospace;font-size:.6rem;
  color:#4a5e78">DAY</span>
<input id="dayN" type="number" value="190" min="0" max="364"
  style="width:60px;background:#04070d;color:#00d4ff;border:1px solid #162240;
  font-family:'IBM Plex Mono',monospace;padding:3px 6px;border-radius:3px">
<button onclick="loadDay()" style="padding:3px 10px;background:#00d4ff;
  color:#04070d;border:none;border-radius:3px;cursor:pointer;
  font-family:'IBM Plex Mono',monospace;font-size:.6rem;font-weight:700">
LOAD</button>
<span id="dayStatus" style="font-family:'IBM Plex Mono',monospace;
  font-size:.55rem;color:#4a5e78;min-width:80px"></span>
</div>
<script>
async function loadDay(){
  const day=+document.getElementById('dayN').value;
  const status=document.getElementById('dayStatus');
  status.textContent='solving day '+day+'...';
  status.style.color='#ffd600';
  try{
    const r=await fetch('/api/sim?day='+day);
    if(!r.ok){
      const e=await r.json().catch(()=>({error:r.statusText}));
      throw new Error(e.error||r.statusText);
    }
    const data=await r.json();
    window.applySimData(data);
    status.textContent='loaded '+(data.date||'day '+day);
    status.style.color='#00e676';
    // Switch to dynamic view automatically since data is now available
    if(typeof setView==='function')setView('dynamic');
  }catch(e){
    status.textContent='error: '+e.message;
    status.style.color='#ff1744';
  }
}
// Auto-load whatever day is in URL params, or 190 by default
window.addEventListener('load',()=>{
  const u=new URLSearchParams(window.location.search);
  if(u.has('day'))document.getElementById('dayN').value=u.get('day');
  loadDay();
});
</script>
"""
    return base.replace("</body>", picker + "</body>")


# ==================================================================
# CLI
# ==================================================================

def main():
    p = argparse.ArgumentParser(
        description="Elermore Vale dashboard v2 — improved")
    p.add_argument("--glm-dir", default="Elermorevale")
    p.add_argument("--common-dir", default="common")
    p.add_argument("--output", default="elermorevale_dashboard_v2.html")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--profiles", default="profiles/fit_profiles.csv")
    p.add_argument("--day", type=int, default=190)
    p.add_argument("--open", action="store_true")
    p.add_argument("--layout", choices=["hierarchical", "spring"],
                   default="hierarchical",
                   help="Layout algorithm (hierarchical preferred)")
    p.add_argument("--serve", action="store_true",
                   help="Run Flask backend instead of static HTML")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--n-monitors", type=int, default=150)
    a = p.parse_args()

    if a.serve:
        run_server(a.glm_dir, a.common_dir, a.profiles,
                   port=a.port, layout=a.layout)
        return

    G, source = build_topology(a.glm_dir)
    sim = None
    if a.simulate:
        sim = run_simulation(a.glm_dir, a.common_dir, a.profiles, a.day,
                             n_monitors=a.n_monitors)
    out = generate_dashboard(G, source, sim, a.output, layout=a.layout)
    if a.open:
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()