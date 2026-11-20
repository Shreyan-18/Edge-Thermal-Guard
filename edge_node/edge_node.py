"""
Edge Thermal Guard — edge_node.py
==================================
Python simulation of an ESP32 thermal edge node.
Runs a producer/consumer threaded pipeline:
  Producer  → captures & preprocesses thermal frames → Queue
  Consumer  → runs ML inference → builds JSON → sends UDP

Usage:
  python edge_node.py --node-id EV-PACK-01 --demo-runaway
"""

import argparse
import json
import math
import queue
import signal
import socket
import sys
import threading
import time
from collections import deque

import numpy as np

# Configure console encoding for UTF-8 emoji support
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ──────────────────────────────────────────────
# CLI ARGS
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Edge Thermal Guard – ESP32 simulator")
    p.add_argument("--node-id",      default="EV-PACK-01",                         help="Node identifier")
    p.add_argument("--host",         default="127.0.0.1",                           help="Dashboard UDP host")
    p.add_argument("--port",         type=int, default=5005,                        help="Dashboard UDP port")
    p.add_argument("--model-path",   default="ml/ml_artifacts/model_int8.tflite",   help="INT8 TFLite model")
    p.add_argument("--sample-grids", default="ml/ml_artifacts/sample_grids.npz",   help="Replay grids (npz)")
    p.add_argument("--interval-ms",  type=int, default=150,                         help="Frame interval ms")
    p.add_argument("--demo-runaway", action="store_true",                           help="Inject runaway ramp after 30 frames")
    return p.parse_args()

# ──────────────────────────────────────────────
# MODEL LOADING & INT8 INTERPRETER
# ──────────────────────────────────────────────
class INT8NumpyInterpreter:
    """
    Lightweight embedded INT8 interpreter that matches TFLite Micro's interface.
    Executes depthwise-separable quantized convolution & integer arithmetic.
    """
    def __init__(self, weights_path):
        data = np.load(weights_path)
        self.q = {k: data[k] for k in data.files}
        self.inp_tensor = None
        self.out_tensor = None

    def get_input_details(self):
        return [{
            "index": 0,
            "dtype": np.uint8,
            "shape": [1, 24, 32, 2],
            "quantization": (1.0 / 255.0, 0),
        }]

    def get_output_details(self):
        return [{
            "index": 1,
            "dtype": np.uint8,
            "shape": [1, 3],
            "quantization": (1.0 / 255.0, 0),
        }]

    def set_tensor(self, idx, tensor):
        self.inp_tensor = tensor

    def invoke(self):
        X = self.inp_tensor.astype(np.float32)
        if self.inp_tensor.dtype == np.uint8:
            X = X / 255.0
        if X.ndim == 3:
            X = X[np.newaxis]
        B, H_in, W_in, _ = X.shape

        x_nchw = np.transpose(X, (0, 3, 1, 2))
        x_pad = np.pad(x_nchw, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")

        w_conv1 = self.q["conv1_w"].astype(np.float32) * float(self.q["conv1_scale"])
        out_conv1 = np.zeros((B, 16, H_in, W_in), dtype=np.float32)
        for f in range(16):
            acc = np.zeros((B, H_in, W_in), dtype=np.float32)
            for c in range(2):
                filt = w_conv1[f, c]
                for kr in range(3):
                    for kc in range(3):
                        acc += filt[kr, kc] * x_pad[:, c, kr:kr+H_in, kc:kc+W_in]
            out_conv1[:, f] = np.maximum(0.0, acc + self.q["conv1_b"][f])

        w_dw = self.q["dw_w"].astype(np.float32) * float(self.q["dw_scale"])
        c1_pad = np.pad(out_conv1, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
        out_dw = np.zeros((B, 16, H_in, W_in), dtype=np.float32)
        for c in range(16):
            dw_f = w_dw[c, 0]
            acc = np.zeros((B, H_in, W_in), dtype=np.float32)
            for kr in range(3):
                for kc in range(3):
                    acc += dw_f[kr, kc] * c1_pad[:, c, kr:kr+H_in, kc:kc+W_in]
            out_dw[:, c] = acc + self.q["dw_b"][c]

        w_pw = self.q["pw_w"].astype(np.float32) * float(self.q["pw_scale"])
        dw_flat = np.transpose(out_dw, (0, 2, 3, 1)).reshape(-1, 16)
        pw_out = (dw_flat @ w_pw.T + self.q["pw_b"]).reshape(B, H_in, W_in, 32)
        out_pw = np.maximum(0.0, np.transpose(pw_out, (0, 3, 1, 2)))

        gap = np.mean(out_pw, axis=(2, 3))
        gmp = np.max(out_pw, axis=(2, 3))
        feats = np.concatenate([gap, gmp], axis=1)

        w_fc1 = self.q["fc1_w"].astype(np.float32) * float(self.q["fc1_scale"])
        fc1_act = np.maximum(0.0, feats @ w_fc1.T + self.q["fc1_b"])

        w_fc2 = self.q["fc2_w"].astype(np.float32) * float(self.q["fc2_scale"])
        logits = fc1_act @ w_fc2.T + self.q["fc2_b"]
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        self.out_tensor = np.clip(np.round(probs * 255.0), 0, 255).astype(np.uint8)

    def get_tensor(self, idx):
        return self.out_tensor


def load_interpreter(model_path):
    """Try tflite_runtime, then tensorflow.lite, then model_weights.npz INT8 engine, else None."""
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        print(f"[edge_node] Model loaded via tflite_runtime: {model_path}", file=sys.stderr)
        return interp
    except Exception:
        pass
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        print(f"[edge_node] Model loaded via tensorflow.lite: {model_path}", file=sys.stderr)
        return interp
    except Exception:
        pass

    # Check for companion model_weights.npz
    from pathlib import Path
    p = Path(model_path)
    weights_path = p.parent / "model_weights.npz"
    if weights_path.exists():
        try:
            interp = INT8NumpyInterpreter(str(weights_path))
            print(f"[edge_node] Model loaded via INT8 quantized engine: {weights_path}", file=sys.stderr)
            return interp
        except Exception as e:
            print(f"[edge_node] Could not load model_weights.npz ({e})", file=sys.stderr)

    print("[edge_node] WARNING: No TFLite runtime found — using heuristic scorer.", file=sys.stderr)
    return None


def heuristic_score(max_temp_c: float, dt_dt: float) -> float:
    """
    Rule-based fallback threat scorer.
    Returns threat score 0-100.
    """
    score = 0.0
    if max_temp_c > 85:
        score += 60
    elif max_temp_c > 60:
        score += 30
    elif max_temp_c > 45:
        score += 10

    if dt_dt > 2.0:
        score += 40
    elif dt_dt > 0.5:
        score += 20

    return min(score, 100.0)


def run_inference(interp, grid: np.ndarray) -> float:
    """
    Run TFLite inference on a (24,32,2) float32 grid.
    Returns threat_score 0-100.
    """
    if interp is None:
        return None  # will use heuristic

    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()

    inp = input_details[0]
    # Determine if model is INT8-quantised
    if inp["dtype"] == np.uint8:
        scale, zero_point = inp["quantization"]
        if scale == 0:
            scale = 1.0 / 255.0
        inp_data = np.clip(
            np.round(grid / scale + zero_point), 0, 255
        ).astype(np.uint8)
    else:
        inp_data = grid.astype(np.float32)

    interp.set_tensor(inp["index"], inp_data[np.newaxis])  # add batch dim
    interp.invoke()

    out = output_details[0]
    raw = interp.get_tensor(out["index"])[0]  # shape (3,)

    # Dequantize if needed
    if out["dtype"] == np.uint8:
        scale, zero_point = out["quantization"]
        if scale == 0:
            scale = 1.0 / 255.0
        raw = (raw.astype(np.float32) - zero_point) * scale

    # Softmax just in case (model output may already be probabilities)
    exp = np.exp(raw - raw.max())
    probs = exp / exp.sum()           # [P(NORMAL), P(WARNING), P(RUNAWAY)]

    score = round(50 * float(probs[1]) + 100 * float(probs[2]))
    return float(score)


# ──────────────────────────────────────────────
# SAMPLE GRID LOADING / SYNTHETIC GENERATION
# ──────────────────────────────────────────────
def load_sample_grids(path: str) -> np.ndarray:
    """Return array of shape (N, 24, 32, 2) float32."""
    try:
        data = np.load(path)
        grids = data["grids"].astype(np.float32)
        print(f"[edge_node] Loaded {len(grids)} sample grids from {path}", file=sys.stderr)
        return grids
    except Exception as e:
        print(f"[edge_node] Could not load sample grids ({e}). Generating synthetic data.", file=sys.stderr)
        return _generate_synthetic_grids(100)


def _generate_synthetic_grids(n: int = 100) -> np.ndarray:
    """Generate (n, 24, 32, 2) synthetic thermal grids (float32, 0-1 range)."""
    rng = np.random.default_rng(42)
    grids = []
    for i in range(n):
        # Channel 0: thermal frame (mostly ambient 20-35 °C → ~0.2-0.35 of 0-100 range)
        base = rng.uniform(0.20, 0.35, (24, 32)).astype(np.float32)
        # Occasional hotspot
        if rng.random() > 0.5:
            cx, cy = rng.integers(4, 28), rng.integers(4, 20)
            temp = rng.uniform(0.45, 0.80)
            base[cy-2:cy+2, cx-2:cx+2] = temp
        ch0 = base
        # Channel 1: dT/dt — small noise
        ch1 = rng.uniform(-0.05, 0.05, (24, 32)).astype(np.float32)
        grids.append(np.stack([ch0, ch1], axis=-1))
    return np.array(grids, dtype=np.float32)


# ──────────────────────────────────────────────
# PRE-PROCESSING
# ──────────────────────────────────────────────
def median_filter_2d(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Simple 2D median filter (no scipy dependency)."""
    pad = size // 2
    out = np.empty_like(arr)
    padded = np.pad(arr, pad, mode="edge")
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            patch = padded[r:r+size, c:c+size]
            out[r, c] = np.median(patch)
    return out


def preprocess(grid: np.ndarray, prev_grid: np.ndarray | None) -> dict:
    """
    grid shape: (24, 32, 2) — values 0-1.
    Returns dict with keys: grid_pp, hotspot_xy, max_temp_c, dt_dt
    """
    ch0 = grid[:, :, 0].copy()

    # Ambient offset: subtract mean of bottom-left 4×4 corner (assumed ambient)
    ambient = ch0[20:24, 0:4].mean()
    ch0 = np.clip(ch0 - ambient + 0.25, 0.0, 1.0)

    # Dead-pixel smoothing (median filter)
    ch0 = median_filter_2d(ch0, size=3)

    # Hotspot
    flat_idx = np.argmax(ch0)
    hy, hx = divmod(int(flat_idx), 32)

    # Max temperature °C (scale 0-100)
    max_temp_c = round(float(ch0.max()) * 100.0, 1)

    # dT/dt
    if prev_grid is not None:
        diff = ch0 - median_filter_2d(prev_grid[:, :, 0], size=3)
        dt_dt = round(float(diff.max()) * 100.0, 2)   # scale back to °C/frame
    else:
        dt_dt = 0.0

    # Rebuild preprocessed grid (channel 0 updated, channel 1 = dt)
    ch1 = np.clip(ch0 - (prev_grid[:, :, 0] if prev_grid is not None else ch0), -0.5, 0.5)
    grid_pp = np.stack([ch0, ch1], axis=-1)

    return {
        "grid_pp":    grid_pp,
        "hotspot_xy": [hx, hy],
        "max_temp_c": max_temp_c,
        "dt_dt":      dt_dt,
    }


# ──────────────────────────────────────────────
# STATE FROM SCORE
# ──────────────────────────────────────────────
def score_to_state(score: int) -> str:
    if score >= 70:
        return "RUNAWAY"
    if score >= 30:
        return "WARNING"
    return "NORMAL"


# ──────────────────────────────────────────────
# SHARED STATE
# ──────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self.running           = True
        self.frames_produced   = 0
        self.frames_consumed   = 0
        self.packets_sent      = 0
        self.udp_fail_streak   = 0
        self.offline_mode      = False
        self.latencies_ms: deque = deque(maxlen=200)
        self.lock              = threading.Lock()


# ──────────────────────────────────────────────
# PRODUCER THREAD
# ──────────────────────────────────────────────
def producer(args, grids: np.ndarray, frame_q: queue.Queue, state: SharedState, demo_ramp_event):
    """
    Reads frames from replay buffer (or synthetic), preprocesses,
    pushes (frame_data, t_capture) into the queue.
    """
    n      = len(grids)
    idx    = 0
    prev_g = None
    seq    = 0
    ramp_frame = 0
    demo_active = False

    while state.running:
        t0 = time.perf_counter()

        # ---- DEMO RUNAWAY INJECTION ----
        if args.demo_runaway and state.frames_produced >= 30:
            if not demo_active:
                demo_active = True
                print("[edge_node] >> DEMO MODE: Injecting thermal runaway ramp <<", file=sys.stderr)

        raw_grid = grids[idx % n].copy()
        idx += 1

        if demo_active:
            # Ramp: inject escalating hotspot over 20 frames
            ramp_frac = min(ramp_frame / 20.0, 1.0)
            hot_temp  = 0.3 + ramp_frac * 0.70   # 0.3 → 1.0 (30°C → 100°C)
            raw_grid[10:14, 14:18, 0] = hot_temp
            ramp_frame += 1

        pp = preprocess(raw_grid, prev_g)
        prev_g = raw_grid

        item = {
            "grid_pp":    pp["grid_pp"],
            "hotspot_xy": pp["hotspot_xy"],
            "max_temp_c": pp["max_temp_c"],
            "dt_dt":      pp["dt_dt"],
            "seq":        seq,
            "t_capture":  time.time(),
        }
        seq += 1

        try:
            frame_q.put(item, timeout=0.5)
        except queue.Full:
            pass  # Drop frame if consumer is too slow

        with state.lock:
            state.frames_produced += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        sleep_s    = max(0.0, args.interval_ms / 1000.0 - elapsed_ms / 1000.0)
        time.sleep(sleep_s)


# ──────────────────────────────────────────────
# CONSUMER THREAD
# ──────────────────────────────────────────────
def consumer(args, interp, sock: socket.socket, state: SharedState):
    """
    Dequeues preprocessed frames, runs inference, sends UDP, prints to stdout.
    """
    while state.running:
        try:
            item = frame_q.get(timeout=0.3)
        except queue.Empty:
            continue

        t_infer_start = time.perf_counter()

        grid_pp    = item["grid_pp"]
        max_temp_c = item["max_temp_c"]
        dt_dt      = item["dt_dt"]
        hx, hy     = item["hotspot_xy"]
        seq        = item["seq"]
        t_capture  = item["t_capture"]

        # ---- INFERENCE ----
        ml_score = run_inference(interp, grid_pp)
        if ml_score is None:
            # Heuristic fallback
            ml_score = heuristic_score(max_temp_c, dt_dt)

        threat_score = int(round(min(max(ml_score, 0), 100)))
        state_str    = score_to_state(threat_score)

        # ---- LOCAL FAILSAFE ----
        failsafe_active = bool(state_str == "RUNAWAY" or max_temp_c > 85.0)
        if failsafe_active:
            print(
                f"🚨 LOCAL FAILSAFE TRIGGERED | node={args.node_id} "
                f"state={state_str} max_c={max_temp_c}°C score={threat_score} seq={seq}",
                file=sys.stderr,
                flush=True,
            )

        # ---- BUILD PACKET ----
        packet = {
            "id":       args.node_id,
            "ts":       round(time.time(), 3),
            "score":    threat_score,
            "max_c":    round(max_temp_c, 1),
            "dt":       round(dt_dt, 2),
            "xy":       [hx, hy],
            "state":    state_str,
            "seq":      seq,
            "failsafe": failsafe_active,
        }
        pkt_bytes = json.dumps(packet).encode()

        # ---- PRINT TO STDOUT ----
        print(json.dumps(packet), flush=True)

        # ---- SEND UDP ----
        sent = False
        if not state.offline_mode:
            try:
                sock.sendto(pkt_bytes, (args.host, args.port))
                sent = True
                with state.lock:
                    state.udp_fail_streak = 0
                    state.packets_sent   += 1
            except Exception:
                with state.lock:
                    state.udp_fail_streak += 1
                    if state.udp_fail_streak >= 3:
                        state.offline_mode = True
                        print("OFFLINE MODE: local failsafe active", file=sys.stderr, flush=True)

        # ---- LATENCY ----
        latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
        with state.lock:
            state.frames_consumed += 1
            state.latencies_ms.append(latency_ms)

        frame_q.task_done()


# ──────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────
def print_stats(state: SharedState, start_time: float):
    elapsed = time.time() - start_time
    avg_lat = (sum(state.latencies_ms) / len(state.latencies_ms)) if state.latencies_ms else 0.0
    print("\n--- Edge Thermal Guard Stats ---", file=sys.stderr)
    print(f"  Node            : {NODE_ID}", file=sys.stderr)
    print(f"  Uptime          : {elapsed:.1f}s", file=sys.stderr)
    print(f"  Frames produced : {state.frames_produced}", file=sys.stderr)
    print(f"  Frames consumed : {state.frames_consumed}", file=sys.stderr)
    print(f"  Packets sent    : {state.packets_sent}", file=sys.stderr)
    print(f"  Avg latency     : {avg_lat:.2f} ms", file=sys.stderr)
    print("--------------------------------", file=sys.stderr)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
NODE_ID   = "EV-PACK-01"   # overridden in main
frame_q: queue.Queue = None


def main():
    global NODE_ID, frame_q

    args    = parse_args()
    NODE_ID = args.node_id

    print(f"[edge_node] Starting node {args.node_id} → {args.host}:{args.port}", file=sys.stderr)
    print(f"[edge_node] Interval: {args.interval_ms}ms | Demo runaway: {args.demo_runaway}", file=sys.stderr)

    # Load model
    interp = load_interpreter(args.model_path)

    # Load grids
    grids = load_sample_grids(args.sample_grids)

    # UDP socket (non-blocking)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    # Shared state
    state = SharedState()
    frame_q = queue.Queue(maxsize=5)
    demo_ramp_event = threading.Event()

    start_time = time.time()

    # SIGINT handler
    def _sigint_handler(sig, frame):
        print("\n[edge_node] Interrupt received — shutting down…", file=sys.stderr)
        state.running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    # Launch threads
    t_prod = threading.Thread(
        target=producer,
        args=(args, grids, frame_q, state, demo_ramp_event),
        daemon=True,
        name="producer",
    )
    t_cons = threading.Thread(
        target=consumer,
        args=(args, interp, sock, state),
        daemon=True,
        name="consumer",
    )
    t_prod.start()
    t_cons.start()

    try:
        while state.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        state.running = False

    # Wait for threads to finish
    t_prod.join(timeout=2.0)
    t_cons.join(timeout=2.0)
    sock.close()

    print_stats(state, start_time)


if __name__ == "__main__":
    main()
