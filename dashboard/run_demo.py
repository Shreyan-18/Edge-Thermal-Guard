"""
Edge Thermal Guard - Demo Launcher
Module 4: Orchestrates collector + edge nodes + dashboard for live demo

Usage:
    python dashboard/run_demo.py
"""

import os
import sys
import time
import signal
import threading
import subprocess
import webbrowser
from pathlib import Path

# ── Determine project root ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

COLLECTOR_SCRIPT = PROJECT_ROOT / "collector" / "udp_collector.py"
EDGE_NODE_SCRIPT = PROJECT_ROOT / "edge_node" / "edge_node.py"
DASHBOARD_SCRIPT = SCRIPT_DIR / "app.py"

DEMO_NODES = [
    {"node_id": "EV-PACK-01", "extra_args": []},
    {"node_id": "EV-PACK-02", "extra_args": []},
    {"node_id": "GRID-XFMR-01", "extra_args": ["--demo-runaway"]},
]

# Track all spawned processes for cleanup
procs = []


def cleanup(sig=None, frame=None):
    """Kill all subprocesses cleanly."""
    print("\n\n[run_demo] Shutting down all processes...", flush=True)
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass

    time.sleep(0.5)

    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass

    print("[run_demo] All processes stopped. Goodbye!", flush=True)
    sys.exit(0)


def stream_output(pipe, label):
    """Drain pipe in background thread to prevent OS buffer deadlocks."""
    try:
        for line in iter(pipe.readline, ''):
            if not line:
                break
            text = line.rstrip()
            # Only print meaningful logs to avoid console clutter
            if any(k in text for k in ("Starting", "Listening", "RUNAWAY", "FAILSAFE", "Registered", "ONLINE", "OFFLINE", "URL:", "http://")):
                print(f"[{label}] {text}", flush=True)
    except Exception:
        pass


def spawn(cmd, label):
    """Spawn a subprocess with non-blocking output streaming."""
    print(f"[run_demo] Starting {label}...", flush=True)
    try:
        env = os.environ.copy()
        # Ensure project root is in PYTHONPATH so certifi and modules load reliably
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        p = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        t = threading.Thread(target=stream_output, args=(p.stdout, label), daemon=True)
        t.start()
        procs.append(p)
        return p
    except Exception as e:
        print(f"[run_demo] WARNING: Could not start {label}: {e}", flush=True)
        return None


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    python = sys.executable

    print("=" * 65, flush=True)
    print("  🔥 EDGE THERMAL GUARD — Complete Live Demo Launcher", flush=True)
    print("=" * 65, flush=True)
    print(f"  Project root : {PROJECT_ROOT}", flush=True)
    print(f"  Python binary: {python}", flush=True)
    print("", flush=True)

    # 1. Start UDP collector
    if COLLECTOR_SCRIPT.exists():
        spawn([python, str(COLLECTOR_SCRIPT)], "COLLECTOR")
    else:
        print(f"[run_demo] WARNING: Collector script not found at {COLLECTOR_SCRIPT}", flush=True)

    time.sleep(1.0)

    # 2. Start edge nodes
    for i, node_cfg in enumerate(DEMO_NODES):
        node_id = node_cfg["node_id"]
        extra = node_cfg["extra_args"]

        cmd = [python, str(EDGE_NODE_SCRIPT), "--node-id", node_id] + extra
        if EDGE_NODE_SCRIPT.exists():
            spawn(cmd, f"EDGE-{node_id}")
        else:
            print(f"[run_demo] WARNING: Edge script not found at {EDGE_NODE_SCRIPT}", flush=True)
            break

        time.sleep(0.4)

    # 3. Wait for initial telemetry
    print("\n[run_demo] Initializing sensor telemetry and neural inference...", flush=True)
    time.sleep(1.5)

    # 4. Start Streamlit dashboard
    dashboard_cmd = [
        python, "-m", "streamlit", "run",
        str(DASHBOARD_SCRIPT),
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--",
        "--demo",
    ]
    spawn(dashboard_cmd, "DASHBOARD")

    time.sleep(2.0)

    print("", flush=True)
    print("=" * 65, flush=True)
    print("  ✅ EDGE THERMAL GUARD IS LIVE!", flush=True)
    print("  🌐 Open Dashboard in your browser: http://localhost:8501", flush=True)
    print("  📡 UDP Telemetry: port 5005 | HTTP API: port 8765", flush=True)
    print("  🚨 GRID-XFMR-01 is configured to ramp into runaway at frame 30", flush=True)
    print("  Press Ctrl+C to safely terminate all nodes.", flush=True)
    print("=" * 65, flush=True)
    print("", flush=True)

    # Attempt to open browser automatically
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    # Keep alive
    try:
        while True:
            time.sleep(5)
            for p in procs:
                if p is not None and p.poll() is not None:
                    rc = p.poll()
                    if rc != 0:
                        print(f"[run_demo] Notice: process exited with code {rc}", flush=True)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
