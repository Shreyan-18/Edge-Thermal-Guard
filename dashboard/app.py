"""
Edge Thermal Guard - Streamlit Dashboard
Module 4: Real-time visualization of decentralized thermal threat detection
"""

import sys
import time
import json
import math
import io
import os
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Edge Thermal Guard",
    layout="wide",
    page_icon="🔥",
)

# ── Parse CLI args (Streamlit passes args after '--') ───────────────────────
def parse_args():
    args = sys.argv[1:]
    if "--" in args:
        args = args[args.index("--") + 1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", action="store_true", default=False)
    parsed, _ = parser.parse_known_args(args)
    return parsed

cli_args = parse_args()
DEMO_FLAG = cli_args.demo

# ── Constants ────────────────────────────────────────────────────────────────
COLLECTOR_URL = "http://127.0.0.1:8765"
NODE_STATES_PATH = Path("collector/node_states.json")
ANOMALY_LOG_PATH = Path("collector/anomaly_log.jsonl")
CONSOLE_FEED_PATH = Path("collector/console_feed.txt")
DEMO_NODES = ["EV-PACK-01", "EV-PACK-02", "GRID-XFMR-01"]
REFRESH_MS = 1000

UDP_PACKET_BYTES = 80
RAW_FRAME_BYTES = 3072

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] { background-color: #161b22 !important; }
[data-testid="stHeader"] { background-color: #0e1117 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}

/* Node cards */
.node-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}
.node-card-failsafe {
    background: #1f1012;
    border: 2px solid #ff4444;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    box-shadow: 0 0 12px rgba(255, 68, 68, 0.25);
}
.node-card-offline {
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    opacity: 0.5;
}

/* Status badges */
.badge-live      { background:#00ff88; color:#000; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.8em; }
.badge-demo      { background:#ff8c00; color:#000; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.8em; }
.badge-normal    { background:#00ff88; color:#000; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-warning   { background:#ffd700; color:#000; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-runaway   { background:#ff4444; color:#fff; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-offline   { background:#555;    color:#ccc; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-failsafe  { background:#ff2222; color:#fff; padding:3px 8px; border-radius:6px; font-size:0.75em; font-weight:700; }

/* Terminal Console Box */
.terminal-box {
    background-color: #0b0f14;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.83em;
    height: 250px;
    overflow-y: auto;
    line-height: 1.6;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.7);
}
.term-line-failsafe {
    color: #ff5555;
    font-weight: 700;
    background-color: rgba(255, 68, 68, 0.14);
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 3px;
    border-left: 3px solid #ff4444;
}
.term-line-warning {
    color: #ffd700;
    background-color: rgba(255, 215, 0, 0.08);
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 3px;
    border-left: 3px solid #ffd700;
}
.term-line-normal {
    color: #00ff88;
    padding: 1px 8px;
    margin-bottom: 2px;
}
.term-line-info {
    color: #79c0ff;
    padding: 1px 8px;
    margin-bottom: 2px;
}

/* Title accent */
.main-title { color: #ff4444; font-size: 2.4em; font-weight: 800; letter-spacing: 1px; }

/* Dataframe */
[data-testid="stDataFrame"] { background: #161b22 !important; }

/* Button styling */
.stButton > button {
    background: #21262d;
    color: #e0e0e0;
    border: 1px solid #30363d;
    border-radius: 6px;
}
.stButton > button:hover { border-color: #ff4444; color: #ff4444; }

/* Footer */
.footer-text { color: #555; font-size: 0.75em; margin-top: 20px; }

/* Progress bar accent */
[data-testid="stProgress"] > div > div { background-color: #00ff88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Auto-refresh ─────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_MS, key="autorefresh")
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ── Session state init ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = {}  # {node_id: deque(maxlen=60)}

if "fault_runaway" not in st.session_state:
    st.session_state["fault_runaway"] = None   # (node_id, expire_ts)

if "fault_blackout" not in st.session_state:
    st.session_state["fault_blackout"] = None  # (node_id, expire_ts)

if "console_logs" not in st.session_state:
    st.session_state["console_logs"] = deque(maxlen=80)

if "thermal_grid_node" not in st.session_state:
    st.session_state["thermal_grid_node"] = "GRID-XFMR-01"

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_from_http():
    """Poll collector's HTTP endpoint."""
    try:
        import requests
        resp = requests.get(COLLECTOR_URL, timeout=0.5)
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get("nodes", data)
            feed = data.get("console_feed", [])
            return nodes, "live", feed
    except Exception:
        pass
    return None, None, []


def fetch_from_file():
    """Read collector/node_states.json as fallback."""
    try:
        if NODE_STATES_PATH.exists():
            with open(NODE_STATES_PATH, "r") as f:
                data = json.load(f)
            nodes = data.get("nodes", data)
            feed = data.get("console_feed", [])
            return nodes, "file", feed
    except Exception:
        pass
    return None, None, []


def generate_demo_nodes():
    """Generate realistic demo data using time-based variation."""
    t = time.time()
    nodes = {}

    runaway_node = None
    blackout_node = None
    now = time.time()

    if st.session_state["fault_runaway"]:
        node_id, expire_ts = st.session_state["fault_runaway"]
        if now < expire_ts:
            runaway_node = node_id
        else:
            st.session_state["fault_runaway"] = None

    if st.session_state["fault_blackout"]:
        node_id, expire_ts = st.session_state["fault_blackout"]
        if now < expire_ts:
            blackout_node = node_id
        else:
            st.session_state["fault_blackout"] = None

    profiles = {
        "EV-PACK-01": {"base": 24, "amp": 10, "freq": 0.05, "phase": 0.0, "max_base": 31.0},
        "EV-PACK-02": {"base": 36, "amp": 16, "freq": 0.03, "phase": 1.2, "max_base": 48.0},
        "GRID-XFMR-01": {"base": 52, "amp": 8, "freq": 0.02, "phase": 2.4, "max_base": 96.5},
    }

    demo_feed = []
    seq_base = int((t * 6.67) % 1000)

    for node_id, p in profiles.items():
        score = p["base"] + p["amp"] * abs(math.sin(t * p["freq"] + p["phase"]))
        score = min(100.0, max(0.0, score + 2.0 * math.sin(t * 0.1 + p["phase"])))

        if node_id == runaway_node:
            score = 96.0 + 4.0 * abs(math.sin(t * 2))

        # GRID-XFMR-01 simulates high-temp failsafe demo (matches user's screenshot: max_c ~96°C)
        if node_id == "GRID-XFMR-01":
            max_temp = 93.0 + 5.8 * abs(math.sin(t * 0.3 + p["phase"]))
            score = 52.0 + 7.0 * abs(math.sin(t * 0.2))
        else:
            max_temp = p["max_base"] + score * 0.4 + 1.5 * math.sin(t * 0.07 + p["phase"])

        dt_dt = 0.1 + (score / 100.0) * 2.5 * abs(math.sin(t * 0.3))
        hotspot_x = int(18 + 5 * math.sin(t * 0.04 + p["phase"]))
        hotspot_y = int(13 + 4 * math.cos(t * 0.04 + p["phase"]))

        if score < 30:
            status = "NORMAL"
        elif score < 70:
            status = "WARNING"
        else:
            status = "RUNAWAY"

        if node_id == blackout_node:
            status = "OFFLINE"

        failsafe_trig = bool(max_temp > 85.0 or status == "RUNAWAY")
        seq_curr = seq_base + hash(node_id) % 20

        nodes[node_id] = {
            "node_id": node_id,
            "threat_score": round(score, 1),
            "score": int(round(score)),
            "status": status,
            "state": status,
            "max_temp_c": round(max_temp, 1),
            "max_c": round(max_temp, 1),
            "dt_dt": round(dt_dt, 3),
            "dt": round(dt_dt, 2),
            "hotspot_x": hotspot_x,
            "hotspot_y": hotspot_y,
            "xy": [hotspot_x, hotspot_y],
            "seq": seq_curr,
            "failsafe": failsafe_trig,
            "failsafe_triggered": failsafe_trig,
            "packets_received": int(t * 6.67) % 10000 + 100,
            "last_seen": t,
            "bandwidth_savings_pct": 97.4,
        }

        # Build simulated log line matching user's terminal
        time_str = time.strftime("%H:%M:%S", time.localtime(t))
        if failsafe_trig:
            demo_feed.append({
                "ts": t,
                "time_str": time_str,
                "node_id": node_id,
                "line": f"[{time_str}] [EDGE-{node_id}] 🚨 LOCAL FAILSAFE TRIGGERED | node={node_id} state={status} max_c={max_temp:.1f}°C score={int(round(score))} seq={seq_curr}",
                "failsafe": True,
                "state": status,
            })
        elif status == "WARNING":
            demo_feed.append({
                "ts": t,
                "time_str": time_str,
                "node_id": node_id,
                "line": f"[{time_str}] [EDGE-{node_id}] ⚠️ THERMAL WARNING | node={node_id} state={status} max_c={max_temp:.1f}°C dt={dt_dt:.2f}°C/s score={int(round(score))} seq={seq_curr}",
                "failsafe": False,
                "state": status,
            })

    bw_savings = 97.4
    return nodes, bw_savings, demo_feed


def fetch_data():
    """Fetch node data from best available source."""
    nodes, source, feed = fetch_from_http()
    if nodes:
        return nodes, source, None, feed

    nodes, source, feed = fetch_from_file()
    if nodes:
        return nodes, source, None, feed

    nodes, bw, demo_feed = generate_demo_nodes()
    return nodes, "demo", bw, demo_feed


def read_anomaly_log(n=20):
    """Read last n entries from anomaly_log.jsonl."""
    entries = []
    try:
        if ANOMALY_LOG_PATH.exists():
            with open(ANOMALY_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass
    except Exception:
        pass
    return entries


# ── 2D Thermal Heatmap Generators ─────────────────────────────────────────────
def generate_thermal_heatmap(hx, hy, max_temp_c, ambient=25.0, node_id=""):
    """
    Generate 24x32 2D thermal array heatmap image buffer with hotspot target.
    Matches MLX90640 (32x24) infrared sensor resolution.
    """
    H, W = 24, 32
    r_coords, c_coords = np.mgrid[0:H, 0:W]
    d2 = (r_coords - hy)**2 + (c_coords - hx)**2
    sigma = 3.5 if max_temp_c < 60 else 5.2
    weight = np.exp(-d2 / (2.0 * sigma**2))

    rng = np.random.default_rng(int(abs(hx) * 13 + abs(hy) * 7 + int(max_temp_c * 10)) % 10000)
    noise = rng.normal(0, 0.35, (H, W))
    grid = ambient + (max_temp_c - ambient) * weight + noise
    grid = np.clip(grid, 18.0, 115.0)

    fig, ax = plt.subplots(figsize=(6.8, 3.8), facecolor="#161b22")
    ax.set_facecolor("#0e1117")
    im = ax.imshow(grid, cmap="inferno", vmin=20, vmax=105, origin="upper", aspect="auto")

    # Crosshair target at hotspot
    ax.plot(hx, hy, marker="+", color="#00ffcc", markersize=14, markeredgewidth=2.2)
    ax.plot(hx, hy, marker="o", color="#00ffcc", markersize=9, fillstyle="none", markeredgewidth=1.8)

    ax.text(
        min(W - 3, max(1, hx + 1.2)),
        max(1.8, min(H - 2, hy - 1.0)),
        f"Hotspot ({hx},{hy})\n{max_temp_c:.1f}°C",
        color="#ffffff",
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#161b22", edgecolor="#ff4444", alpha=0.9)
    )

    ax.set_title(f"Channel 0: 2D Thermal Array ({node_id}) — 32x24 IR Sensor Grid", color="#e0e0e0", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Sensor Columns (X: 0..31)", color="#888888", fontsize=8.5)
    ax.set_ylabel("Sensor Rows (Y: 0..23)", color="#888888", fontsize=8.5)
    ax.tick_params(colors="#888888", labelsize=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Temperature (°C)", color="#e0e0e0", fontsize=8.5)
    cb.ax.tick_params(colors="#e0e0e0", labelsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def generate_dt_heatmap(hx, hy, dt_dt, node_id=""):
    """Generate 24x32 2D dT/dt differential heatmap image buffer."""
    H, W = 24, 32
    r_coords, c_coords = np.mgrid[0:H, 0:W]
    d2 = (r_coords - hy)**2 + (c_coords - hx)**2
    sigma = 4.0
    weight = np.exp(-d2 / (2.0 * sigma**2))

    rng = np.random.default_rng(int(abs(hx) * 7 + abs(hy)) % 10000)
    noise = rng.normal(0, 0.05, (H, W))
    grid = dt_dt * weight + noise
    grid = np.clip(grid, -2.0, max(5.0, dt_dt * 1.2))

    fig, ax = plt.subplots(figsize=(6.8, 3.8), facecolor="#161b22")
    ax.set_facecolor("#0e1117")
    im = ax.imshow(grid, cmap="coolwarm", vmin=-1.0, vmax=max(3.0, dt_dt), origin="upper", aspect="auto")

    ax.plot(hx, hy, marker="+", color="#ffffff", markersize=14, markeredgewidth=2.0)

    ax.set_title(f"Channel 1: 2D dT/dt Rate of Rise ({node_id})", color="#e0e0e0", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Sensor Columns (X: 0..31)", color="#888888", fontsize=8.5)
    ax.set_ylabel("Sensor Rows (Y: 0..23)", color="#888888", fontsize=8.5)
    ax.tick_params(colors="#888888", labelsize=8)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("dT/dt Rate (°C/s)", color="#e0e0e0", fontsize=8.5)
    cb.ax.tick_params(colors="#e0e0e0", labelsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ── Plotly helpers ────────────────────────────────────────────────────────────
def make_gauge(score, node_id):
    """Create a Plotly Indicator gauge for threat score."""
    try:
        import plotly.graph_objects as go

        if score < 30:
            bar_color = "#00ff88"
            border_color = "#00aa55"
        elif score < 70:
            bar_color = "#ffd700"
            border_color = "#cc9900"
        else:
            bar_color = "#ff4444"
            border_color = "#cc0000"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "", "font": {"size": 28, "color": "#e0e0e0"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#555",
                    "tickfont": {"color": "#888", "size": 10},
                },
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "#161b22",
                "borderwidth": 2,
                "bordercolor": border_color,
                "steps": [
                    {"range": [0, 30],  "color": "#0a2a1a"},
                    {"range": [30, 70], "color": "#2a2200"},
                    {"range": [70, 100],"color": "#2a0a0a"},
                ],
                "threshold": {
                    "line": {"color": "#ff4444", "width": 3},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
            title={"text": "Threat Score", "font": {"size": 13, "color": "#888"}},
        ))
        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            margin=dict(l=20, r=20, t=40, b=10),
            height=200,
            font={"color": "#e0e0e0"},
        )
        return fig
    except ImportError:
        return None


def make_timeseries(history):
    """Create a Plotly line chart of score history for all nodes."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        palette = {
            "EV-PACK-01":  "#00ff88",
            "EV-PACK-02":  "#ffd700",
            "GRID-XFMR-01": "#ff4444",
        }
        default_colors = ["#00bfff", "#ff69b4", "#da70d6", "#ffa500"]
        idx = 0

        for node_id, dq in history.items():
            if not dq:
                continue
            times = [entry[0] for entry in dq]
            scores = [entry[1] for entry in dq]
            if times:
                t0 = times[-1]
                rel_times = [t - t0 for t in times]
            else:
                rel_times = []

            color = palette.get(node_id, default_colors[idx % len(default_colors)])
            idx += 1

            fig.add_trace(go.Scatter(
                x=rel_times,
                y=scores,
                mode="lines",
                name=node_id,
                line={"color": color, "width": 2},
                hovertemplate="%{y:.1f}<extra>" + node_id + "</extra>",
            ))

        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            font={"color": "#e0e0e0"},
            xaxis={
                "title": "Seconds ago",
                "gridcolor": "#21262d",
                "zerolinecolor": "#30363d",
                "tickformat": ".0f",
            },
            yaxis={
                "title": "Threat Score",
                "range": [0, 100],
                "gridcolor": "#21262d",
                "zerolinecolor": "#30363d",
            },
            legend={"bgcolor": "#161b22", "bordercolor": "#30363d", "borderwidth": 1},
            margin=dict(l=50, r=20, t=30, b=40),
            height=280,
            shapes=[
                dict(type="line", y0=70, y1=70, x0=0, x1=1, xref="paper",
                     line=dict(color="#ff4444", width=1, dash="dash")),
                dict(type="line", y0=30, y1=30, x0=0, x1=1, xref="paper",
                     line=dict(color="#00ff88", width=1, dash="dot")),
            ],
        )
        return fig
    except ImportError:
        return None


# ── Main render ───────────────────────────────────────────────────────────────
nodes, data_source, bw_savings_override, incoming_console_feed = fetch_data()

now_ts = time.time()

# Ingest console feed into session_state deque
if incoming_console_feed:
    for item in incoming_console_feed:
        line_str = item if isinstance(item, str) else item.get("line", str(item))
        if line_str and (not st.session_state["console_logs"] or line_str != st.session_state["console_logs"][-1]):
            st.session_state["console_logs"].append(line_str)

# Also check for console_feed.txt written by collector
if CONSOLE_FEED_PATH.exists():
    try:
        with open(CONSOLE_FEED_PATH, "r", encoding="utf-8") as f:
            c_lines = f.readlines()
        for cl in c_lines[-40:]:
            s_cl = cl.strip()
            if s_cl and s_cl not in st.session_state["console_logs"]:
                st.session_state["console_logs"].append(s_cl)
    except Exception:
        pass

# Update score history
if nodes:
    for node_id, node in nodes.items():
        if node_id not in st.session_state["history"]:
            st.session_state["history"][node_id] = deque(maxlen=60)
        score = node.get("threat_score", node.get("score", 0))
        if node.get("status") != "OFFLINE":
            st.session_state["history"][node_id].append((now_ts, score))

# ── HEADER ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([6, 2])
with col_title:
    st.markdown('<span class="main-title">🔥 Edge Thermal Guard</span>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if data_source == "live":
        st.markdown('<span class="badge-live">● LIVE UDP STREAM</span>', unsafe_allow_html=True)
    elif data_source == "file":
        st.markdown('<span class="badge-live">● STATE FILE LINK</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-demo">● DEMO SIMULATION</span>', unsafe_allow_html=True)

st.markdown("---")

# ── SUMMARY METRICS ───────────────────────────────────────────────────────────
if nodes:
    online_nodes = [n for n in nodes.values() if n.get("status") != "OFFLINE"]
    critical_alerts = sum(1 for n in online_nodes if n.get("status") == "RUNAWAY")
    failsafe_count = sum(
        1 for n in online_nodes
        if bool(n.get("failsafe") or n.get("failsafe_triggered") or n.get("max_temp_c", n.get("max_c", 0)) > 85.0 or n.get("status") == "RUNAWAY")
    )

    # Bandwidth savings
    if bw_savings_override is not None:
        bw_pct = bw_savings_override
    else:
        bw_values = [n.get("bandwidth_savings_pct", 97.4) for n in online_nodes]
        bw_pct = sum(bw_values) / len(bw_values) if bw_values else 97.4

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("🖥️ Nodes Online", f"{len(online_nodes)} / {len(nodes)}")
    with mc2:
        delta_color = "inverse" if critical_alerts > 0 else "normal"
        st.metric("🚨 Critical Alerts", critical_alerts,
                  delta=f"+{critical_alerts}" if critical_alerts else None,
                  delta_color=delta_color)
    with mc3:
        if failsafe_count > 0:
            st.metric("⚡ Failsafe Cutoffs", f"{failsafe_count} ACTIVE", delta="RELAY OPEN (85°C+)", delta_color="inverse")
        else:
            st.metric("⚡ Failsafe Cutoffs", "0 Active", delta="Standby")
    with mc4:
        st.metric("📡 Bandwidth Saved", f"{bw_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PER-NODE CARDS ────────────────────────────────────────────────────────
    st.subheader("🔬 Node Hardware Telemetry & Failsafe Status")
    node_list = list(nodes.values())
    num_cols = min(len(node_list), 3)
    cols = st.columns(num_cols)

    for i, node in enumerate(node_list):
        col = cols[i % num_cols]
        with col:
            nid = node.get("node_id", node.get("id", f"Node-{i}"))
            status = node.get("status", node.get("state", "NORMAL"))
            score = float(node.get("threat_score", node.get("score", 0)))
            max_temp = float(node.get("max_temp_c", node.get("max_c", 0)))
            dt_dt = float(node.get("dt_dt", node.get("dt", 0)))
            xy = node.get("xy", [node.get("hotspot_x", 0), node.get("hotspot_y", 0)])
            hx = xy[0] if isinstance(xy, (list, tuple)) and len(xy) > 0 else node.get("hotspot_x", 0)
            hy = xy[1] if isinstance(xy, (list, tuple)) and len(xy) > 1 else node.get("hotspot_y", 0)
            pkts = int(node.get("packets_received", node.get("packet_count", 0)))
            seq_num = int(node.get("seq", pkts))
            is_failsafe = bool(node.get("failsafe") or node.get("failsafe_triggered") or max_temp > 85.0 or status == "RUNAWAY")

            card_class = "node-card-failsafe" if is_failsafe else ("node-card-offline" if status == "OFFLINE" else "node-card")

            if status == "OFFLINE":
                st.markdown(f"""
                <div class="node-card-offline">
                  <h4>⚫ {nid}</h4>
                  <span class="badge-offline">⚫ OFFLINE</span>
                  <p style="color:#555; margin-top:8px;">No signal received (Heartbeat timeout)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                if status == "RUNAWAY":
                    badge = '<span class="badge-runaway">🔴 RUNAWAY</span>'
                elif status == "WARNING":
                    badge = '<span class="badge-warning">🟡 WARNING</span>'
                else:
                    badge = '<span class="badge-normal">🟢 NORMAL</span>'

                st.markdown(f"""
                <div class="{card_class}">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0;">{nid}</h4>
                    {badge}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Failsafe relay banner
                if is_failsafe:
                    st.markdown(f"""
                    <div style="background:#2d0f0f; border:2px solid #ff4444; border-radius:6px; padding:6px; margin:4px 0 10px 0; text-align:center;">
                        <span style="color:#ff4444; font-weight:800; font-size:0.9em;">🚨 LOCAL FAILSAFE TRIGGERED</span><br>
                        <span style="color:#ffb3b3; font-size:0.75em;">Hardware Cutoff Relay ACTIVE &nbsp;|&nbsp; seq={seq_num}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#0c2415; border:1px solid #00aa55; border-radius:6px; padding:4px; margin:4px 0 10px 0; text-align:center;">
                        <span style="color:#00ff88; font-size:0.75em; font-weight:600;">🛡️ Hardware Failsafe: Standby (Relay Closed)</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Gauge
                fig_gauge = make_gauge(score, nid)
                if fig_gauge:
                    st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{nid}")
                else:
                    bar_col = "#00ff88" if score < 30 else ("#ffd700" if score < 70 else "#ff4444")
                    st.markdown(f"""
                    <div style="background:#161b22; border-radius:8px; padding:10px; margin:8px 0; border:1px solid #30363d;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="color:#888; font-size:0.85em;">Threat Score</span>
                            <span style="color:{bar_col}; font-size:1.3em; font-weight:700;">{score:.0f} / 100</span>
                        </div>
                        <div style="background:#21262d; border-radius:6px; height:10px; overflow:hidden;">
                            <div style="background:{bar_col}; width:{min(100.0, max(0.0, score))}%; height:100%; transition:width 0.3s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Stats
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | 🌡️ Max Temp | **{max_temp:.1f} °C** |
                | ⚡ dT/dt | **{dt_dt:.3f} °C/s** |
                | 📍 Hotspot | **({hx}, {hy})** |
                | 🔢 Packet Seq | **seq={seq_num}** |
                | 📦 Total Packets | **{pkts:,}** |
                """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2D THERMAL ARRAY INPUT INSPECTOR ──────────────────────────────────────
    st.subheader("🔬 2D Thermal Array Input Inspector (Sensor Grids Fed to AI)")
    st.caption("Inspect the raw 24×32 2-channel thermal matrices (MLX90640 IR array) analyzed locally by the quantized INT8 CNN.")

    active_nids = [n.get("node_id", n.get("id")) for n in node_list if n.get("status") != "OFFLINE"]
    if not active_nids:
        active_nids = list(DEMO_NODES)

    sel_node_id = st.selectbox(
        "Select Edge Node to View 2D Input Array:",
        active_nids,
        index=active_nids.index("GRID-XFMR-01") if "GRID-XFMR-01" in active_nids else 0,
        key="sel_node_heatmap"
    )

    sel_node_data = next((n for n in node_list if n.get("node_id") == sel_node_id or n.get("id") == sel_node_id), node_list[0])
    s_hx = sel_node_data.get("hotspot_x", 18)
    s_hy = sel_node_data.get("hotspot_y", 13)
    s_max_c = float(sel_node_data.get("max_temp_c", sel_node_data.get("max_c", 95.0)))
    s_dt = float(sel_node_data.get("dt_dt", sel_node_data.get("dt", 2.5)))
    s_seq = sel_node_data.get("seq", 0)

    # Render heatmaps in 2 columns (Channel 0 & Channel 1)
    hm_col1, hm_col2 = st.columns(2)
    with hm_col1:
        img_ch0 = generate_thermal_heatmap(s_hx, s_hy, s_max_c, ambient=25.0, node_id=sel_node_id)
        st.image(img_ch0, use_container_width=True)
    with hm_col2:
        img_ch1 = generate_dt_heatmap(s_hx, s_hy, s_dt, node_id=sel_node_id)
        st.image(img_ch1, use_container_width=True)

    # Thermal array details banner
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 14px; margin-top:4px; font-size:0.85em;">
      <strong>Sensor Grid Specs:</strong> Resolution: <code>32×24 (768 IR pixels)</code> &nbsp;|&nbsp;
      Input Tensor Shape: <code>(24, 32, 2)</code> &nbsp;|&nbsp;
      Detected Hotspot: <strong style="color:#00ffcc;">({s_hx}, {s_hy})</strong> &nbsp;|&nbsp;
      Peak Temperature: <strong style="color:#ff5555;">{s_max_c:.1f}°C</strong> &nbsp;|&nbsp;
      Thermal Acceleration: <strong style="color:#ffd700;">{s_dt:.2f}°C/s</strong> &nbsp;|&nbsp;
      Frame Seq: <code>seq={s_seq}</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LIVE EDGE CONSOLE & FAILSAFE STREAM ────────────────────────────────────
    st.subheader("💻 Live Edge Hardware Stream & Failsafe Console Feed")
    st.caption("Live streaming event telemetry broadcast from embedded nodes (matches hardware console output).")

    term_col1, term_col2 = st.columns([5, 1])
    with term_col1:
        filter_node = st.selectbox(
            "Filter Console Output by Node:",
            ["All Active Nodes"] + list(active_nids),
            key="console_filter"
        )
    with term_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 Clear Feed"):
            st.session_state["console_logs"].clear()

    # Build HTML lines for terminal
    raw_logs = list(st.session_state["console_logs"])
    if not raw_logs:
        # Populate with initial events if newly opened
        t_now_str = time.strftime("%H:%M:%S")
        raw_logs = [
            f"[{t_now_str}] [EDGE-GRID-XFMR-01] 🚨 LOCAL FAILSAFE TRIGGERED | node=GRID-XFMR-01 state=WARNING max_c=96.9°C score=52 seq=331",
            f"[{t_now_str}] [EDGE-GRID-XFMR-01] 🚨 LOCAL FAILSAFE TRIGGERED | node=GRID-XFMR-01 state=WARNING max_c=97.9°C score=52 seq=332",
            f"[{t_now_str}] [EDGE-GRID-XFMR-01] 🚨 LOCAL FAILSAFE TRIGGERED | node=GRID-XFMR-01 state=WARNING max_c=95.9°C score=52 seq=333",
            f"[{t_now_str}] [EDGE-EV-PACK-01]   📡 TELEMETRY INGEST | node=EV-PACK-01 state=NORMAL max_c=31.2°C dt=0.45°C/s score=22 seq=334",
            f"[{t_now_str}] [EDGE-EV-PACK-02]   ⚠️ THERMAL WARNING | node=EV-PACK-02 state=WARNING max_c=46.5°C dt=1.82°C/s score=45 seq=335",
        ]

    # Filter lines
    filtered_lines = []
    for l in raw_logs:
        if filter_node == "All Active Nodes" or filter_node in l:
            filtered_lines.append(l)

    # Format HTML
    html_lines = []
    for line in filtered_lines[-35:]:
        # Clean up any escaped unicode characters
        line_clean = line.replace("\\U0001f6a8", "🚨").replace("\\u0001f6a8", "🚨")
        if "FAILSAFE TRIGGERED" in line_clean or "RUNAWAY" in line_clean:
            html_lines.append(f'<div class="term-line-failsafe">{line_clean}</div>')
        elif "WARNING" in line_clean:
            html_lines.append(f'<div class="term-line-warning">{line_clean}</div>')
        elif "TELEMETRY" in line_clean or "NORMAL" in line_clean:
            html_lines.append(f'<div class="term-line-normal">{line_clean}</div>')
        else:
            html_lines.append(f'<div class="term-line-info">{line_clean}</div>')

    terminal_html = f"""
    <div class="terminal-box" id="terminal-scroll">
      {"".join(html_lines) if html_lines else '<div style="color:#555;">No events matching filter...</div>'}
    </div>
    """
    st.markdown(terminal_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TIME-SERIES CHART ─────────────────────────────────────────────────────
    st.subheader("📈 Live Threat Score History (last 60s)")
    fig_ts = make_timeseries(st.session_state["history"])
    if fig_ts:
        st.plotly_chart(fig_ts, use_container_width=True, key="timeseries")
    else:
        import pandas as pd
        chart_data = {}
        for k_nid, dq in st.session_state["history"].items():
            if dq:
                chart_data[k_nid] = [entry[1] for entry in dq]
        if chart_data:
            maxlen = max(len(v) for v in chart_data.values())
            for k_nid, v in chart_data.items():
                if len(v) < maxlen:
                    chart_data[k_nid] = [v[0]] * (maxlen - len(v)) + v
            df_chart = pd.DataFrame(chart_data)
            st.line_chart(df_chart, height=260)
        else:
            st.info("Awaiting telemetry stream...")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ANOMALY LOG ───────────────────────────────────────────────────────────
    st.subheader("📋 Anomaly Log (last 20 events)")
    log_entries = read_anomaly_log(20)
    if log_entries:
        import pandas as pd
        df = pd.DataFrame(log_entries)

        display_cols = [c for c in ["ts", "node_id", "state", "status", "score",
                                     "max_c", "max_temp_c", "dt", "seq", "failsafe"] if c in df.columns]
        if display_cols:
            df_display = df[display_cols].copy()
        else:
            df_display = df

        def highlight_status(row):
            status_val = str(row.get("state", row.get("status", "")))
            if "RUNAWAY" in status_val or row.get("failsafe") is True:
                return ["background-color: #2a0a0a; color: #ff4444"] * len(row)
            elif "WARNING" in status_val:
                return ["background-color: #2a2200; color: #ffd700"] * len(row)
            return ["background-color: #0a2a1a; color: #00ff88"] * len(row)

        try:
            styled = df_display.style.apply(highlight_status, axis=1)
            st.dataframe(styled, use_container_width=True, height=260)
        except Exception:
            st.dataframe(df_display, use_container_width=True, height=260)
    else:
        st.info("No anomaly log entries yet. Start the collector to see events.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── EDGE vs CLOUD COMPARISON ──────────────────────────────────────────────
    st.subheader("⚡ Edge vs Cloud Bandwidth Comparison")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("""
        **📡 Edge Node (UDP Packet)**
        - Size: **~80 bytes**
        - Contents: threat score, max temp, hotspot coords, dT/dt
        - Inference: **on-device** (TFLite INT8)
        - Latency: **< 5 ms**
        """)
    with ec2:
        st.markdown("""
        **☁️ Cloud Approach (Raw Frame Stream)**
        - Size: **3,072 bytes** (32×24 float32 array)
        - Contents: raw pixel temperature array
        - Inference: cloud server round-trip
        - Latency: **300 - 3000 ms**
        """)

    st.progress(min(1.0, bw_pct / 100.0))
    st.write(f"**Bandwidth reduction: {bw_pct:.1f}%** (transmitting ~80B instead of 3,072B per frame)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DEMO CONTROLS ─────────────────────────────────────────────────────────
    if data_source == "demo" or DEMO_FLAG:
        st.subheader("🎮 Demo Controls")
        st.caption("Inject faults to demonstrate system response")
        dc1, dc2, dc3 = st.columns(3)

        with dc1:
            if st.button("🔥 Inject Thermal Runaway"):
                target = node_list[0].get("node_id", DEMO_NODES[0]) if node_list else DEMO_NODES[0]
                st.session_state["fault_runaway"] = (target, time.time() + 10)
                st.toast(f"🔥 Thermal runaway injected on {target} (10s)", icon="🔥")

        with dc2:
            if st.button("📡 Simulate Wi-Fi Blackout"):
                target = node_list[1].get("node_id", DEMO_NODES[1]) if len(node_list) > 1 else DEMO_NODES[0]
                st.session_state["fault_blackout"] = (target, time.time() + 8)
                st.toast(f"📡 Wi-Fi blackout on {target} (8s)", icon="📡")

        with dc3:
            if st.button("✅ Reset All"):
                st.session_state["fault_runaway"] = None
                st.session_state["fault_blackout"] = None
                st.toast("✅ All faults cleared", icon="✅")

else:
    st.warning("⏳ Waiting for collector...")
    st.markdown("""
    **To get started:**
    1. Run the UDP collector: `python collector/udp_collector.py`
    2. Run an edge node: `python edge_node/edge_node.py --node-id EV-PACK-01`
    3. Or launch demo: `python dashboard/run_demo.py`
    """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
source_label = {
    "live": "🟢 HTTP connected (port 8765)",
    "file": "🟡 File fallback (collector/node_states.json)",
    "demo": "🟠 Demo mode (simulated data)",
}.get(data_source, "⚫ No data source")

st.markdown(
    f'<div class="footer-text">Last updated: {time.strftime("%Y-%m-%d %H:%M:%S")} UTC &nbsp;|&nbsp; '
    f'Collector: {source_label} &nbsp;|&nbsp; '
    f'Edge Thermal Guard v1.0</div>',
    unsafe_allow_html=True,
)

# Native auto-refresh loop when streamlit_autorefresh is not installed
if not HAS_AUTOREFRESH:
    time.sleep(1.0)
    st.rerun()
