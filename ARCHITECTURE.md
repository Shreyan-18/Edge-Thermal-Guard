# 🏗️ Edge Thermal Guard — Architecture Deep Dive

> This document describes the technical internals of every component. For setup and demo instructions see [README.md](README.md).

---

## 1. System Overview

Edge Thermal Guard is a **four-tier edge-computing pipeline** for real-time thermal anomaly detection. The key design philosophy is:

> **"Classify at the source. Transmit conclusions, not data."**

```
┌────────────┐  raw     ┌──────────────────────┐  UDP 72B ┌──────────────┐  HTTP ┌───────────┐
│ IR Sensor  │─────────▶│  ESP32 (TFLite Micro) │─────────▶│  Collector   │──────▶│ Dashboard │
│ 32×24 px   │  frame   │  INT8 CNN + preproc   │  alert   │  REST + Log  │  poll │ Streamlit │
└────────────┘          └──────────────────────┘           └──────────────┘       └───────────┘
   3,072 B/frame              ≤ 10 ms latency                ring buffer             live UI
```

---

## 2. Data Flow Walkthrough

| Step | Actor | Action | Output |
|------|--------|--------|--------|
| 1 | IR Sensor | Captures 32×24 thermal frame at up to 32 Hz | Raw float array (°C) |
| 2 | ESP32 Preprocessor | Median-filters dead pixels, corrects ambient offset | Cleaned frame |
| 3 | ESP32 Preprocessor | Computes dT/dt: `delta = frame[t] - frame[t-1]` | Rate-of-change frame |
| 4 | ESP32 Preprocessor | Stacks channels → shape `(24, 32, 2)`, normalizes 0–1 | Input tensor |
| 5 | ESP32 INT8 CNN | Runs TFLite Micro inference | `[P_normal, P_warn, P_runaway]` |
| 6 | ESP32 Scorer | Computes threat score (see formula below) | Integer 0–100 |
| 7 | ESP32 Sender | Serializes 72-byte JSON UDP packet | UDP datagram |
| 8 | Collector UDP Listener | Receives packet, validates schema | Parsed dict |
| 9 | Collector Ring Buffer | Non-blocking enqueue to 1,000-slot ring | Buffered record |
| 10 | Collector REST Handler | Dequeues, updates node state, writes anomaly log | JSONL record |
| 11 | Dashboard Poller | HTTP GET `/nodes` every 1 s | JSON node list |
| 12 | Dashboard Renderer | Updates heatmap, score timeline, alert cards | Streamlit UI |

---

## 3. INT8 Quantization

### Why Quantize?

Standard TensorFlow trains with **float32** weights (4 bytes each). On an ESP32:
- Flash: 4 MB total; OS + libraries use ~1.5 MB; leaving ~2.5 MB for model
- SRAM: 320–512 KB; a float32 tensor arena of 40 KB leaves little for the app

INT8 post-training quantization maps each float32 weight to an 8-bit integer using:

```
x_int8 = clamp(round(x_fp32 / scale) + zero_point, -128, 127)
```

Where:
- `scale = (x_max - x_min) / 255`
- `zero_point = round(-x_min / scale) - 128`

### Size & Speed Comparison

| Metric | FP32 Model | INT8 Model | Improvement |
|--------|-----------|-----------|-------------|
| Model size | ~72 KB | ~18 KB | **4× smaller** |
| Tensor arena RAM | ~40 KB | ~10 KB | **4× less RAM** |
| Inference latency (ESP32) | ~32 ms | ~8 ms | **4× faster** |
| Top-1 accuracy | 97.2% | 96.5% | **−0.7% drop** |
| Flash requirement | 72 KB | 18 KB | Fits ESP32 easily |

### Calibration Process

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8
```

The `representative_dataset_gen` feeds ~100 real training samples so TFLite can measure per-layer activation ranges and choose optimal `scale` / `zero_point` pairs.

---

## 4. Non-Blocking Pipeline Design

Both the edge node simulator and the collector use a **producer-consumer ring buffer** to decouple time-critical tasks:

```
Producer Thread                    Consumer Thread
──────────────                     ───────────────
  Inference / UDP recv               UDP send / REST serve
       │                                    │
       ▼                                    ▼
  ┌─────────────────────────────────────────────┐
  │   Ring Buffer  (collections.deque, maxlen=N) │
  │   [ slot0 | slot1 | slot2 | ... | slotN ]   │
  └─────────────────────────────────────────────┘
  Thread-safe via threading.Lock               ^
  Enqueue: O(1) append                         │
  Dequeue: O(1) popleft                        │
  On overflow: oldest slot silently evicted ───┘
```

**Why this matters:** A blocking send call during a Wi-Fi retransmit storm would stall inference. With the ring buffer, inference always proceeds at full speed; the sender drains independently.

---

## 5. UDP Packet Format

### JSON Schema (72 bytes typical)

```json
{
  "node_id":     "EV-PACK-01",
  "ts":          1700000000.123,
  "threat_score": 42,
  "label":       "WARNING",
  "max_temp_c":  61.3,
  "hotspot_x":   18,
  "hotspot_y":   11,
  "dt_dt":       4.2,
  "p_normal":    0.12,
  "p_warning":   0.76,
  "p_runaway":   0.12,
  "seq":         1042
}
```

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | string | Unique ESP32 identifier |
| `ts` | float | Unix timestamp (seconds) |
| `threat_score` | int 0–100 | Composite danger score |
| `label` | enum | NORMAL / WARNING / RUNAWAY |
| `max_temp_c` | float | Peak pixel temperature °C |
| `hotspot_x/y` | int | Pixel coords of hotspot |
| `dt_dt` | float | Max dT/dt in °C/s across frame |
| `p_*` | float 0–1 | Softmax probabilities (sum to 1) |
| `seq` | int | Monotonic packet counter |

### Binary Alternative (future optimization)

For ultra-constrained links (LoRa, 802.15.4), replace JSON with a 20-byte binary struct:

```
 0       1       2       3
 ┌───────┬───────┬───────┬───────┐
 │ node  │ score │ label │ seqhi │
 │ hash  │ uint8 │ uint8 │ uint8 │
 ├───────┴───────┴───────┴───────┤
 │        timestamp uint32       │
 ├───────────────────────────────┤
 │       max_temp_c float16      │
 │         dt_dt   float16       │
 ├───────────────────────────────┤
 │    hotspot_x    hotspot_y     │
 │      uint8        uint8       │
 └───────────────────────────────┘
  Total: 20 bytes vs 72 bytes JSON  → 72% further reduction
```

---

## 6. Threat Score Formula

```
threat_score = round(50 × P(WARNING) + 100 × P(RUNAWAY))
```

| Score Range | Interpretation | Dashboard Color |
|---|---|---|
| 0–24 | NORMAL — no anomaly | 🟢 Green |
| 25–59 | WARNING — elevated heat | 🟡 Yellow |
| 60–100 | RUNAWAY — critical alert | 🔴 Red |

**Rationale:** P(RUNAWAY) is weighted at 100 (full scale) because runaway is the terminal, dangerous state. P(WARNING) is weighted at 50 to give meaningful signal without false alarms. NORMAL contributes 0 so low-confidence NORMAL predictions don't mask real signals.

---

## 7. Edge vs Cloud Comparison

| Dimension | Cloud-Based Monitoring | Edge Thermal Guard |
|-----------|----------------------|-------------------|
| Inference location | Remote GPU server | ESP32 on-device |
| Latency (detect → alert) | 100–500 ms | **< 10 ms** |
| Bandwidth per node @ 10 Hz | ~92 KB/s (raw frames) | **~0.7 KB/s** (alerts) |
| Bandwidth savings | — | **98.5%** |
| Works offline | ❌ No | **✅ Yes (failsafe)** |
| Privacy (raw thermal data) | Leaves premises | **Never leaves device** |
| Infrastructure cost | Cloud GPU + egress | ESP32 + local PC |
| Scalability | Limited by bandwidth | Linear with nodes |
| Single point of failure | Cloud server | None (distributed) |

---

## 8. ESP32 Memory Budget

| Resource | Allocation | Used | Remaining |
|----------|-----------|------|-----------|
| Flash (4 MB total) | App + libs | 1.5 MB | 2.5 MB |
| Flash — TFLite model | model_int8.tflite | **18 KB** | 2.48 MB |
| SRAM (320 KB total) | FreeRTOS + stack | 80 KB | 240 KB |
| SRAM — Tensor arena | Input/output buffers | **10 KB** | 230 KB |
| SRAM — Ring buffer | 16 × 20-byte slots | 0.3 KB | 230 KB |
| SRAM — Wi-Fi stack | lwIP + TLS | 60 KB | 170 KB |
| SRAM — Application | Variables + heap | 30 KB | **140 KB free** |

**Tensor Arena breakdown (10 KB):**
- Input tensor `(1, 24, 32, 2)` uint8 = 1,536 bytes
- Conv layer scratch = 4,096 bytes
- Depthwise scratch = 2,048 bytes
- GAP + Dense = 512 bytes
- Output `(1, 3)` uint8 = 3 bytes
- **Total: ~8.2 KB** (fits in 10 KB arena with margin)

---

## 9. Offline Failsafe State Machine

```
                     ┌─────────────────┐
              ┌─────▶│   CONNECTED     │◀──────────────┐
              │      │  Normal ops:    │               │
              │      │  UDP send OK    │               │
              │      └────────┬────────┘               │
              │               │ send() fails            │
              │               │ OR socket timeout       │
              │               ▼                         │
              │      ┌─────────────────┐               │
              │      │  DISCONNECTED   │               │
              │      │  Continue local │               │
              │      │  inference      │               │
              │      │  Buffer alerts  │               │
              │      │  to flash/file  │               │
              │      └────────┬────────┘               │
              │               │ periodic reconnect      │
              │               │ attempt (every 5s)      │
              │               ▼                         │
              │      ┌─────────────────┐               │
              │      │  RECONNECTING   │               │
              │      │  Test UDP send  │               │
              │      │  to collector   │               │
              │      └────────┬────────┘               │
              │               │ success                 │
              │               ▼                         │
              │      ┌─────────────────┐               │
              └──────│  REPLAYING      │───────────────┘
                     │  Drain buffered │  replay complete
                     │  alerts in seq  │
                     │  order → UDP    │
                     └─────────────────┘

  Flash buffer: circular, max 1,000 records (20 KB @ 20 B/record binary)
  Replay: at most 50 records/second to avoid flooding collector
```

---

## 10. Security Considerations

| Threat | Risk | Mitigation |
|--------|------|-----------|
| UDP spoofing | Fake node injects false alerts | HMAC-SHA256 `sig` field (optional, pre-shared key per node) |
| Replay attack | Attacker replays old high-score packets | Monotonic `seq` counter; reject out-of-order |
| Denial of service | Flood collector with UDP | Rate-limit per source IP (100 pkt/s); drop excess |
| Data exfiltration | Raw thermal frames leave premises | Never transmitted — only scores leave the edge |
| Physical tampering | Attacker modifies ESP32 firmware | Secure boot + flash encryption (ESP32-S3 feature) |
| Collector REST | Unauthenticated API | Bind to localhost only; add JWT for production |

> [!NOTE]
> For a hackathon demo, the HMAC and JWT features are optional. The network is assumed to be a trusted LAN. For production deployment, enable all mitigations.

> [!CAUTION]
> Never expose the collector REST API (`port 5006`) to the public internet without authentication. It exposes live facility thermal data.
