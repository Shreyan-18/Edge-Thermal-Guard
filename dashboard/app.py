"""
Edge Thermal Guard - Streamlit Dashboard
Module 4: Real-time visualization of decentralized thermal threat detection
"""

import sys
import time
import json
import math
import argparse
from collections import deque
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Edge Thermal Guard",
    layout="wide",
    page_icon="🔥",
)

# ── Parse CLI args (Streamlit passes args after '--') ───────────────────────
def parse_args():
    args = sys.argv[1:]
    # strip streamlit's own flags
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
.node-card-offline {
    background: #1a1a1a;
    border: 1px solid #444;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
    opacity: 0.5;
}

/* Status badges */
.badge-live   { background:#00ff88; color:#000; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.8em; }
.badge-demo   { background:#ff8c00; color:#000; padding:3px 10px; border-radius:12px; font-weight:700; font-size:0.8em; }
.badge-normal  { background:#00ff88; color:#000; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-warning { background:#ffd700; color:#000; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-runaway { background:#ff4444; color:#fff; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }
.badge-offline { background:#555;    color:#ccc; padding:2px 8px; border-radius:8px; font-size:0.75em; font-weight:600; }

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

# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_from_http():
    """Poll collector's HTTP endpoint."""
    try:
        import requests
        resp = requests.get(COLLECTOR_URL, timeout=0.5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("nodes", data), "live"
    except Exception:
        pass
    return None, None


def fetch_from_file():
    """Read collector/node_states.json as fallback."""
    try:
        if NODE_STATES_PATH.exists():
            with open(NODE_STATES_PATH, "r") as f:
                data = json.load(f)
            return data.get("nodes", data), "file"
    except Exception:
        pass
    return None, None


def generate_demo_nodes():
    """Generate realistic demo data using time-based variation."""
    t = time.time()
    nodes = {}

    # Apply injected faults
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
        "EV-PACK-01": {"base": 22, "amp": 15, "freq": 0.05, "phase": 0.0},
        "EV-PACK-02": {"base": 35, "amp": 25, "freq": 0.03, "phase": 1.2},
        "GRID-XFMR-01": {"base": 55, "amp": 20, "freq": 0.02, "phase": 2.4},
    }

    for node_id, p in profiles.items():
        score = p["base"] + p["amp"] * abs(math.sin(t * p["freq"] + p["phase"]))
        score = min(100, max(0, score + 3 * math.sin(t * 0.1 + p["phase"])))

        if node_id == runaway_node:
            score = 95 + 5 * abs(math.sin(t * 2))

        max_temp = 25 + score * 0.7 + 2 * math.sin(t * 0.07 + p["phase"])
        dt_dt = 0.1 + (score / 100) * 2.5 * abs(math.sin(t * 0.3))
        hotspot_x = int(16 + 8 * math.sin(t * 0.04 + p["phase"]))
        hotspot_y = int(12 + 6 * math.cos(t * 0.04 + p["phase"]))

        if score < 30:
            status = "NORMAL"
        elif score < 70:
            status = "WARNING"
        else:
            status = "RUNAWAY"

        if node_id == blackout_node:
            status = "OFFLINE"

        nodes[node_id] = {
            "node_id": node_id,
            "threat_score": round(score, 1),
            "status": status,
            "max_temp_c": round(max_temp, 1),
            "dt_dt": round(dt_dt, 3),
            "hotspot_x": hotspot_x,
            "hotspot_y": hotspot_y,
            "packets_received": int(t * 6.67) % 10000 + 100,
            "last_seen": t,
            "bandwidth_savings_pct": 97.4,
        }

    bw_savings = 97.4  # (1 - 80/3072)*100
    return nodes, bw_savings


def fetch_data():
    """Fetch node data from best available source."""
    nodes, source = fetch_from_http()
    if nodes:
        return nodes, source, None

    nodes, source = fetch_from_file()
    if nodes:
        return nodes, source, None

    if DEMO_FLAG or True:  # Always fall back to demo if no live data
        nodes, bw = generate_demo_nodes()
        return nodes, "demo", bw

    return None, None, None


def read_anomaly_log(n=20):
    """Read last n entries from anomaly_log.jsonl."""
    entries = []
    try:
        if ANOMALY_LOG_PATH.exists():
            with open(ANOMALY_LOG_PATH, "r") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    pass
    except Exception:
        pass
    return entries


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
            # Relative seconds
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
nodes, data_source, bw_savings_override = fetch_data()

now_ts = time.time()

# Update score history
if nodes:
    for node_id, node in nodes.items():
        if node_id not in st.session_state["history"]:
            st.session_state["history"][node_id] = deque(maxlen=60)
        score = node.get("threat_score", 0)
        if node.get("status") != "OFFLINE":
            st.session_state["history"][node_id].append((now_ts, score))

# ── HEADER ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([6, 1])
with col_title:
    st.markdown('<span class="main-title">🔥 Edge Thermal Guard</span>', unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if data_source == "live":
        st.markdown('<span class="badge-live">● LIVE</span>', unsafe_allow_html=True)
    elif data_source == "file":
        st.markdown('<span class="badge-live">● FILE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-demo">● DEMO</span>', unsafe_allow_html=True)

st.markdown("---")

# ── SUMMARY METRICS ───────────────────────────────────────────────────────────
if nodes:
    online_nodes = [n for n in nodes.values() if n.get("status") != "OFFLINE"]
    critical_alerts = sum(1 for n in online_nodes if n.get("status") == "RUNAWAY")

    # Bandwidth savings
    if bw_savings_override is not None:
        bw_pct = bw_savings_override
    else:
        bw_values = [n.get("bandwidth_savings_pct", 97.4) for n in online_nodes]
        bw_pct = sum(bw_values) / len(bw_values) if bw_values else 97.4

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("🖥️ Nodes Online", f"{len(online_nodes)} / {len(nodes)}")
    with mc2:
        delta_color = "inverse" if critical_alerts > 0 else "normal"
        st.metric("🚨 Critical Alerts", critical_alerts,
                  delta=f"+{critical_alerts}" if critical_alerts else None,
                  delta_color=delta_color)
    with mc3:
        st.metric("📡 Bandwidth Saved", f"{bw_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PER-NODE CARDS ────────────────────────────────────────────────────────
    st.subheader("🔬 Node Status")
    node_list = list(nodes.values())
    # Up to 3 columns
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

            if status == "OFFLINE":
                st.markdown(f"""
                <div class="node-card-offline">
                  <h4>⚫ {nid}</h4>
                  <span class="badge-offline">⚫ OFFLINE</span>
                  <p style="color:#555; margin-top:8px;">No signal received</p>
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
                <div class="node-card">
                  <h4 style="margin:0 0 4px 0;">{nid}</h4>
                  {badge}
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
                | 🌡️ Max Temp | {max_temp:.1f} °C |
                | ⚡ dT/dt | {dt_dt:.3f} °C/s |
                | 📍 Hotspot | ({hx}, {hy}) |
                | 📦 Packets | {pkts:,} |
                """)

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

        # Ensure useful columns are present
        display_cols = [c for c in ["timestamp", "node_id", "status", "threat_score",
                                     "max_temp_c", "dt_dt", "message"] if c in df.columns]
        if display_cols:
            df_display = df[display_cols].copy()
        else:
            df_display = df

        # Color-code by status
        def highlight_status(row):
            status_val = str(row.get("status", ""))
            if "RUNAWAY" in status_val:
                return ["background-color: #2a0a0a; color: #ff4444"] * len(row)
            elif "WARNING" in status_val:
                return ["background-color: #2a2200; color: #ffd700"] * len(row)
            return ["background-color: #0a2a1a; color: #00ff88"] * len(row)

        try:
            styled = df_display.style.apply(highlight_status, axis=1)
            st.dataframe(styled, use_container_width=True, height=300)
        except Exception:
            st.dataframe(df_display, use_container_width=True, height=300)
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
        **☁️ Cloud-only Approach (Raw Frame)**
        - Size: **3072 bytes** (32×24×4 float32)
        - Full thermal grid upload every cycle
        - Inference: **in cloud** (requires connectivity)
        - Latency: **50–500 ms**
        """)

    savings_pct = (1 - UDP_PACKET_BYTES / RAW_FRAME_BYTES) * 100
    st.markdown(f"**Live Bandwidth Savings: {savings_pct:.1f}%**")
    st.progress(int(savings_pct) / 100)
    st.caption(f"Sending {UDP_PACKET_BYTES}B instead of {RAW_FRAME_BYTES}B per inference cycle")

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
    # No data at all
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
