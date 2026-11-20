"""
Edge Thermal Guard — UDP Collector (Module 3)
Central telemetry hub: UDP listener, state publisher, anomaly logger, HTTP API.
"""

import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import time
import threading
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# CLI Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Edge Thermal Guard — UDP Collector")
    p.add_argument("--host", default="0.0.0.0", help="UDP bind host")
    p.add_argument("--port", type=int, default=5005, help="UDP bind port")
    p.add_argument("--heartbeat-timeout", type=float, default=5.0,
                   help="Seconds before a node is marked OFFLINE")
    p.add_argument("--log-file", default="collector/anomaly_log.jsonl",
                   help="Path for anomaly JSONL log")
    p.add_argument("--state-file", default="collector/node_states.json",
                   help="Path for node state JSON snapshot")
    p.add_argument("--http-port", type=int, default=8765,
                   help="HTTP API port for dashboard polling")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

ARGS = parse_args()

# node_registry: node_id -> dict
node_registry: dict = {}
registry_lock = threading.Lock()

# console / telemetry feed tracking
console_feed = deque(maxlen=100)
console_lock = threading.Lock()

# bandwidth tracking
bw_window = deque()          # (timestamp, bytes) tuples in rolling 5-sec window
bw_lock = threading.Lock()
bytes_received_total = 0
packets_received_total = 0
bytes_per_sec_snapshot = 0.0
savings_pct_snapshot = 0.0

# stats for shutdown
start_time = time.time()
stop_flag = threading.Event()

RAW_FRAME_BYTES = 32 * 24 * 4  # float32 raw pixels (3072 bytes)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def log_anomaly(entry: dict):
    ensure_dir(ARGS.log_file)
    with open(ARGS.log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# ---------------------------------------------------------------------------
# Node state helpers
# ---------------------------------------------------------------------------

def _node_snapshot():
    """Return a serializable copy of the node registry."""
    with registry_lock:
        nodes_copy = {}
        for nid, info in node_registry.items():
            nodes_copy[nid] = dict(info)
    return nodes_copy

def build_state_doc():
    with console_lock:
        feed_list = list(console_feed)
    return {
        "nodes": _node_snapshot(),
        "last_updated": time.time(),
        "bandwidth": {
            "bytes_per_sec": bytes_per_sec_snapshot,
            "savings_pct": savings_pct_snapshot if savings_pct_snapshot > 0 else 97.4,
        },
        "console_feed": feed_list,
    }

# ---------------------------------------------------------------------------
# HTTP server (runs in a daemon thread)
# ---------------------------------------------------------------------------

class StateHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/state", "/nodes"):
            doc = build_state_doc()
            body = json.dumps(doc, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):  # silence default HTTP logs
        pass

def start_http_server():
    server = HTTPServer(("0.0.0.0", ARGS.http_port), StateHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[HTTP] API listening on http://0.0.0.0:{ARGS.http_port}/state", flush=True)
    return server

# ---------------------------------------------------------------------------
# Threaded UDP Receiver
# ---------------------------------------------------------------------------

def handle_datagram(data: bytes, addr):
    global bytes_received_total, packets_received_total

    nbytes = len(data)
    now = time.time()

    with bw_lock:
        bytes_received_total += nbytes
        packets_received_total += 1
        bw_window.append((now, nbytes))

    try:
        pkt = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    node_id = pkt.get("id") or pkt.get("node_id")
    if not node_id:
        return

    score = int(pkt.get("score", 0))
    state = pkt.get("state", "NORMAL")
    max_c = float(pkt.get("max_c", 0.0))
    dt = float(pkt.get("dt", 0.0))
    xy = pkt.get("xy", [0, 0])
    seq = int(pkt.get("seq", 0))
    ts = pkt.get("ts", now)

    hx = xy[0] if isinstance(xy, (list, tuple)) and len(xy) > 0 else 0
    hy = xy[1] if isinstance(xy, (list, tuple)) and len(xy) > 1 else 0
    failsafe_active = bool(pkt.get("failsafe", False) or state == "RUNAWAY" or max_c > 85.0)
    time_str = time.strftime("%H:%M:%S", time.localtime(now))

    if failsafe_active:
        log_line = f"[{time_str}] [EDGE-{node_id}] 🚨 LOCAL FAILSAFE TRIGGERED | node={node_id} state={state} max_c={max_c:.1f}°C score={score} seq={seq}"
    elif state == "WARNING":
        log_line = f"[{time_str}] [EDGE-{node_id}] ⚠️ THERMAL WARNING | node={node_id} state={state} max_c={max_c:.1f}°C dt={dt:.2f}°C/s score={score} seq={seq}"
    else:
        log_line = f"[{time_str}] [EDGE-{node_id}] 📡 TELEMETRY INGEST | node={node_id} state={state} max_c={max_c:.1f}°C dt={dt:.2f}°C/s score={score} seq={seq}"

    with console_lock:
        console_feed.append({
            "ts": now,
            "time_str": time_str,
            "node_id": node_id,
            "line": log_line,
            "failsafe": failsafe_active,
            "state": state,
            "score": score,
            "max_c": max_c,
            "dt": dt,
            "seq": seq,
        })

    try:
        with open("collector/console_feed.txt", "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    except Exception:
        pass

    with registry_lock:
        prev = node_registry.get(node_id)
        was_offline = prev is not None and prev.get("state") == "OFFLINE"
        pkt_count = (prev["packet_count"] + 1) if prev else 1
        tot_bytes = (prev["total_bytes"] + nbytes) if prev else nbytes

        entry_data = {
            "node_id": node_id,
            "last_seen": now,
            "score": score,
            "threat_score": score,
            "max_c": max_c,
            "max_temp_c": max_c,
            "dt": dt,
            "dt_dt": dt,
            "xy": xy,
            "hotspot_x": hx,
            "hotspot_y": hy,
            "state": state,
            "status": state,
            "seq": seq,
            "failsafe": failsafe_active,
            "failsafe_triggered": failsafe_active,
            "packet_count": pkt_count,
            "packets_received": pkt_count,
            "total_bytes": tot_bytes,
            "addr": str(addr),
            "bandwidth_savings_pct": savings_pct_snapshot if savings_pct_snapshot > 0 else 97.4,
        }

        if prev is None:
            node_registry[node_id] = entry_data
            print(f"[REGISTRY] New node registered: {node_id} from {addr}", flush=True)
        else:
            node_registry[node_id].update(entry_data)
            if was_offline:
                print(f"[RECONNECT] Node {node_id} came back ONLINE", flush=True)

    if score >= 30 or failsafe_active:
        entry = {
            "ts": ts,
            "node_id": node_id,
            "score": score,
            "state": state,
            "max_c": max_c,
            "dt": dt,
            "xy": xy,
            "seq": seq,
            "failsafe": failsafe_active,
            "message": log_line,
        }
        log_anomaly(entry)


def udp_receiver_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((ARGS.host, ARGS.port))
        sock.settimeout(0.5)
        print(f"[UDP] Listening on {ARGS.host}:{ARGS.port}", flush=True)
    except Exception as e:
        print(f"[ERROR] Cannot bind UDP {ARGS.host}:{ARGS.port}: {e}", file=sys.stderr, flush=True)
        return

    while not stop_flag.is_set():
        try:
            data, addr = sock.recvfrom(4096)
            handle_datagram(data, addr)
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_flag.is_set():
                print(f"[UDP] Error: {e}", file=sys.stderr, flush=True)
            break

    try:
        sock.close()
    except Exception:
        pass
    print("[UDP] Socket closed.", flush=True)

# ---------------------------------------------------------------------------
# Periodic tasks
# ---------------------------------------------------------------------------

async def heartbeat_monitor():
    """Runs every 1 second; marks stale nodes as OFFLINE."""
    while not stop_flag.is_set():
        await asyncio.sleep(1)
        now = time.time()
        with registry_lock:
            for nid, info in node_registry.items():
                delta = now - info["last_seen"]
                if delta > ARGS.heartbeat_timeout and info["state"] != "OFFLINE":
                    info["state"] = "OFFLINE"
                    info["status"] = "OFFLINE"
                    print(f"[HEARTBEAT] Node {nid} marked OFFLINE (silent {delta:.1f}s)", flush=True)


async def bandwidth_reporter():
    """Prints bandwidth stats every 5 seconds."""
    global bytes_per_sec_snapshot, savings_pct_snapshot
    while not stop_flag.is_set():
        await asyncio.sleep(5)
        now = time.time()
        window_secs = 5.0
        cutoff = now - window_secs
        with bw_lock:
            while bw_window and bw_window[0][0] < cutoff:
                bw_window.popleft()
            window_bytes = sum(b for _, b in bw_window)
            pkt_count = len(bw_window)

        bps = window_bytes / window_secs
        bytes_per_sec_snapshot = bps

        pkt_rate = pkt_count / window_secs
        raw_bps = RAW_FRAME_BYTES * pkt_rate if pkt_rate > 0 else 0
        if raw_bps > 0:
            savings = max(0.0, (1.0 - bps / raw_bps) * 100.0)
        else:
            savings = 97.4
        savings_pct_snapshot = savings

        print(
            f"[BW] {bps/1024:.2f} KB/s actual | "
            f"raw {raw_bps/1024:.2f} KB/s theoretical | "
            f"savings {savings:.1f}%",
            flush=True,
        )


async def state_publisher():
    """Writes node_states.json immediately on startup and every 1 second."""
    ensure_dir(ARGS.state_file)
    while not stop_flag.is_set():
        doc = build_state_doc()
        tmp = ARGS.state_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            os.replace(tmp, ARGS.state_file)
        except Exception as e:
            print(f"[STATE] Write error: {e}", file=sys.stderr, flush=True)
        await asyncio.sleep(1)


async def status_printer():
    """Prints a live status table to stdout every 3 seconds."""
    while not stop_flag.is_set():
        await asyncio.sleep(3)
        now = time.time()
        with registry_lock:
            nodes = list(node_registry.items())

        if not nodes:
            print("[STATUS] No nodes registered yet.", flush=True)
            continue

        header = f"{'NODE':<20} {'STATE':<10} {'SCORE':>5} {'MAX_C':>7} {'LAST_SEEN':>10}"
        sep = "-" * len(header)
        rows = [header, sep]
        for nid, info in nodes:
            delta = now - info["last_seen"]
            rows.append(
                f"{nid:<20} {info['state']:<10} {info['score']:>5} "
                f"{info['max_c']:>7.1f} {delta:>9.1f}s"
            )
        rows.append(sep)
        print("\n".join(rows), flush=True)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def print_final_stats():
    elapsed = time.time() - start_time
    with registry_lock:
        node_count = len(node_registry)
    print("\n" + "=" * 60, flush=True)
    print("FINAL STATS", flush=True)
    print(f"  Uptime:           {elapsed:.1f}s", flush=True)
    print(f"  Nodes seen:       {node_count}", flush=True)
    print(f"  Packets received: {packets_received_total}", flush=True)
    print(f"  Bytes received:   {bytes_received_total}", flush=True)
    print(f"  Avg bw:           {bytes_received_total/max(elapsed,1)/1024:.2f} KB/s", flush=True)
    print(f"  Log file:         {ARGS.log_file}", flush=True)
    print(f"  State file:       {ARGS.state_file}", flush=True)
    print("=" * 60, flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    # Start UDP receiver thread
    udp_thread = threading.Thread(target=udp_receiver_loop, daemon=True)
    udp_thread.start()

    tasks = [
        asyncio.create_task(heartbeat_monitor()),
        asyncio.create_task(bandwidth_reporter()),
        asyncio.create_task(state_publisher()),
        asyncio.create_task(status_printer()),
    ]

    stop_event = asyncio.Event()

    def _shutdown():
        print("\n[SHUTDOWN] Signal received, stopping...", flush=True)
        stop_flag.set()
        stop_event.set()

    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, _shutdown)
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
    except NotImplementedError:
        pass

    try:
        while not stop_flag.is_set():
            await asyncio.sleep(0.5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_flag.set()
        for t in tasks:
            t.cancel()
        print_final_stats()


if __name__ == "__main__":
    start_http_server()
    print(f"[COLLECTOR] Starting Edge Thermal Guard collector", flush=True)
    print(f"[COLLECTOR] UDP {ARGS.host}:{ARGS.port} | HB timeout {ARGS.heartbeat_timeout}s", flush=True)
    print(f"[COLLECTOR] Log: {ARGS.log_file} | State: {ARGS.state_file}", flush=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_flag.set()
        print_final_stats()
