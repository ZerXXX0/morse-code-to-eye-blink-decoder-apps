import cv2
import json
import asyncio
import hashlib
import os
import secrets
import sqlite3
import sys
import threading
from typing import Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Resolve paths relative to this file so the server works regardless of CWD
# (e.g. `python backend/server.py` from the project root).
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
DB_PATH = os.path.join(BACKEND_DIR, "app.db")

# Import sistem utama dari file implementation.py
from implementation import (
    EyeBlinkMorseSystem,
    SystemConfig,
    CalibrationData,
    CalibrationMethod,
)

app = FastAPI()

# Mencegah error CORS dari React (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi sistem AI dengan path absolut agar tidak bergantung pada CWD.
config = SystemConfig(
    yolo_model_path=os.path.join(BACKEND_DIR, "runs", "classify", "nano_100", "weights", "best.pt"),
)
# MediaPipe face_landmarker.task is downloaded into CWD by default; make sure
# implementation.py resolves it next to itself by chdir-ing into BACKEND_DIR
# for the duration of process startup.
os.chdir(BACKEND_DIR)
system = EyeBlinkMorseSystem(config)

# Enable NLP by default; load IndoBERT in the background to avoid blocking startup.
system.nlp_manager.enable()

def _preload_nlp():
    try:
        system.nlp_manager._ensure_corrector_loaded()
        print("LOG: IndoBERT NLP model loaded")
    except Exception as e:
        print(f"LOG: NLP preload failed: {e}")

threading.Thread(target=_preload_nlp, daemon=True).start()

# Variabel Global
latest_frame = None
latest_data = {}
is_camera_running = False

# Concurrency primitives.
# db_lock     — serializes sqlite writes across the camera thread and request handlers.
# system_lock — guards swaps of the shared CalibrationData on login (the camera
#               thread reads/mutates the same instance during calibration).
db_lock = threading.Lock()
system_lock = threading.Lock()

# EAR baseline collection — camera thread fills _ear_samples while mode is set.
_ear_collect_lock = threading.Lock()
_ear_collect_mode: Optional[str] = None   # 'open' | 'closed' | None
_ear_samples: list = []
_ear_target_count: int = 45               # 3 s at 15 fps
_ear_done_event = threading.Event()

# Tracks who is currently logged in so the calibration-save callback knows
# which user_id to write under. None means "no user is using the app right now."
active_user_id: Optional[int] = None

# --- SQLITE INIT ---
# check_same_thread=False because the camera thread also writes via the
# calibration callback. All writes go through db_lock.
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_conn.row_factory = sqlite3.Row
with db_lock:
    db_conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS calibrations (
            user_id INTEGER PRIMARY KEY,
            data_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS calibration_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )
    db_conn.commit()


def hash_password(password: str, salt: str) -> str:
    """pbkdf2_hmac SHA-256, 100k iterations. Stdlib-only and rainbow-table resistant."""
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    return secrets.compare_digest(hash_password(password, salt), stored_hash)


def serialize_calibration(cal: CalibrationData) -> str:
    """Defensive float casts protect against numpy scalars sneaking in from EAR math."""
    return json.dumps({
        "is_calibrated": bool(cal.is_calibrated),
        "avg_blink_duration_ms": float(cal.avg_blink_duration_ms),
        "avg_dot_duration_ms": float(cal.avg_dot_duration_ms),
        "avg_dash_duration_ms": float(cal.avg_dash_duration_ms),
        "dot_durations": [float(x) for x in cal.dot_durations],
        "dash_durations": [float(x) for x in cal.dash_durations],
        "ear_baseline_open": float(cal.ear_baseline_open),
        "ear_baseline_closed": float(cal.ear_baseline_closed),
        "calibration_method": cal.calibration_method.value,
    })


def deserialize_calibration(data_json: str) -> CalibrationData:
    d = json.loads(data_json)
    return CalibrationData(
        is_calibrated=d["is_calibrated"],
        avg_blink_duration_ms=d["avg_blink_duration_ms"],
        avg_dot_duration_ms=d["avg_dot_duration_ms"],
        avg_dash_duration_ms=d["avg_dash_duration_ms"],
        dot_durations=list(d["dot_durations"]),
        dash_durations=list(d["dash_durations"]),
        ear_baseline_open=d["ear_baseline_open"],
        ear_baseline_closed=d["ear_baseline_closed"],
        calibration_method=CalibrationMethod(d["calibration_method"]),
    )


def save_active_calibration(cal: CalibrationData):
    """Wired into CalibrationManager.on_calibration_complete. Runs on the
    camera thread when finalization happens, persisting under the currently
    logged-in user_id."""
    uid = active_user_id
    if uid is None:
        print("LOG: calibration finalized but no active user — skipping persist")
        return
    payload = serialize_calibration(cal)
    with db_lock:
        db_conn.execute(
            """
            INSERT INTO calibrations (user_id, data_json, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(user_id) DO UPDATE SET data_json=excluded.data_json, updated_at=datetime('now')
            """,
            (uid, payload),
        )
        db_conn.commit()
    print(f"LOG: persisted calibration for user_id={uid}")


system.set_calibration_complete_callback(save_active_calibration)


def _require_user_id(request: Request) -> int:
    """Read X-User-Id header; raise 401/400 on missing/invalid."""
    raw = request.headers.get("X-User-Id")
    if not raw:
        raise HTTPException(status_code=401, detail="missing X-User-Id header")
    try:
        return int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid X-User-Id")

def camera_worker():
    """Single long-lived thread. Opens the camera once at startup, then
    processes frames while is_camera_running is True and idles otherwise.
    This eliminates per-start driver initialisation delay entirely."""
    global latest_frame, latest_data, _ear_collect_mode, _ear_samples
    import time as _time

    # Open the camera once — this is the only time VideoCapture is constructed.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"LOG: Camera opened (isOpened={cap.isOpened()})")

    target_interval = 1.0 / config.target_fps

    while True:  # run for the lifetime of the server process
        if not is_camera_running:
            latest_frame = None
            _time.sleep(0.05)  # idle at ~20 Hz while stopped
            continue

        frame_start = _time.monotonic()

        ret, frame = cap.read()
        if not ret:
            _time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)

        annotated_frame, results = system.process_frame(frame, enable_detection=True)

        # EAR baseline collection for calibration
        ear_val = results.get('ear', 0.0)
        with _ear_collect_lock:
            if _ear_collect_mode is not None and ear_val > 0.01:
                _ear_samples.append(ear_val)
                if len(_ear_samples) >= _ear_target_count:
                    _ear_collect_mode = None
                    _ear_done_event.set()

        cal = system.calibration_manager.get_calibration()
        latest_data = {
            "eyeState": results['eye_state'].value.upper(),
            "confidence": results['confidence'],
            "fps": round(system.blink_detector.estimated_fps, 1),
            "morseSequence": results['morse_sequence'],
            "decodedText": results['decoded_text'],
            "nlpText": results.get('nlp_smoothed_text', ''),
            "isCalibrating": results.get('is_calibrating', False),
            "calProgress": results.get('calibration_progress', ("DONE", 0, 0)),
            "isCalibrated": cal.is_calibrated,
            "calDotMs": round(cal.avg_dot_duration_ms, 1),
            "calDashMs": round(cal.avg_dash_duration_ms, 1),
            "calThresholdMs": round(cal.avg_blink_duration_ms, 1),
            "ear": round(ear_val, 4),
            "nlpSentences": system.nlp_manager.get_structured_sentences(),
        }

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        latest_frame = buffer.tobytes()

        elapsed = _time.monotonic() - frame_start
        if elapsed < target_interval:
            _time.sleep(target_interval - elapsed)


# Start the worker once at module load — it idles until is_camera_running is set.
threading.Thread(target=camera_worker, daemon=True).start()

# --- ENDPOINT KONTROL KAMERA ---

@app.post("/api/camera/start")
async def start_camera_api():
    global is_camera_running
    is_camera_running = True
    return {"status": "camera started"}

@app.post("/api/camera/stop")
async def stop_camera_api():
    global is_camera_running
    is_camera_running = False
    return {"status": "camera stopped"}

@app.post("/api/camera/reset")
async def reset_camera_api():
    system.clear_text() 
    return {"status": "system reset"}

# --- ENDPOINT VIDEO & DATA ---

@app.get("/video_feed")
def video_feed():
    """Endpoint untuk stream gambar ke tag <img> di React.

    Stays open for the lifetime of the HTTP connection. Only yields when a
    NEW frame is ready — otherwise the loop would re-send the same JPEG
    thousands of times per second and freeze the browser tab.
    """
    import time

    def generate_frames():
        last_frame = None
        while True:
            frame = latest_frame  # atomic ref read (GIL)
            if frame is not None and frame is not last_frame:
                last_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Poll ~50 Hz — fast enough to keep up with a 30 FPS source,
                # slow enough to not pin a CPU core.
                time.sleep(0.02)

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/data")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket untuk data real-time."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(latest_data)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("React Client Disconnected")

# --- ENDPOINT KONFIGURASI AI ---

@app.post("/api/clear_text")
def clear_text():
    system.clear_text()
    return {"status": "cleared"}

@app.post("/api/update_config")
async def update_config(data: dict):
    # Forward every recognised SystemConfig field. Unknown keys are silently ignored
    # by EyeBlinkMorseSystem.update_config via its hasattr check.
    allowed = {
        "alpha", "blink_threshold",
        "letter_gap_seconds", "word_gap_seconds", "sentence_gap_seconds",
        "ear_min", "ear_max",
        "smoothing_window", "ema_alpha",
        "use_gpu",
    }
    payload = {k: v for k, v in data.items() if k in allowed}
    system.update_config(**payload)
    return {"status": "updated", "applied": payload}

@app.post("/api/toggle_nlp")
async def toggle_nlp():
    enabled = system.toggle_nlp()
    return {"enabled": enabled}

@app.post("/api/start_calibration")
async def start_calibration(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    blink_count = max(1, min(int(body.get("blink_count", 3)), 20))
    system.start_calibration(target_blinks=blink_count)
    return {"status": "started", "blink_count": blink_count}

@app.post("/api/next_step")
async def next_step():
    if hasattr(system, 'next_calibration_step'):
        system.next_calibration_step()
    return {"status": "next_step_triggered"}

@app.post("/api/reset_calibration")
async def reset_calibration():
    system.reset_calibration()
    return {"status": "reset"}

# --- AUTH ENDPOINTS ---

@app.post("/api/auth/signup")
async def signup(data: dict):
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")
    salt = secrets.token_hex(16)
    pw_hash = hash_password(password, salt)
    try:
        with db_lock:
            cur = db_conn.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, pw_hash, salt),
            )
            db_conn.commit()
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already taken")
    return {"user_id": user_id, "username": username, "has_calibration": False}


@app.post("/api/auth/login")
async def login(data: dict):
    """Verifies credentials, swaps in the user's stored calibration if any,
    and marks them as the active_user_id for subsequent saves."""
    global active_user_id

    if is_camera_running:
        # The camera thread mutates calibration state mid-loop. Force a clean
        # handoff before swapping users.
        raise HTTPException(status_code=409, detail="stop the camera before switching users")

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    with db_lock:
        row = db_conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None or not verify_password(password, row["salt"], row["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid credentials")

    user_id = row["id"]

    # Look up stored calibration.
    with db_lock:
        cal_row = db_conn.execute(
            "SELECT data_json FROM calibrations WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    has_calibration = cal_row is not None
    with system_lock:
        if has_calibration:
            try:
                cal = deserialize_calibration(cal_row["data_json"])
                system.apply_calibration(cal)
                print(f"LOG: loaded calibration for user_id={user_id}")
            except Exception as e:
                # Corrupt row — fall back to a fresh state but don't fail login.
                print(f"failed to load calibration for user_id={user_id}: {e}")
                system.reset_calibration()
                has_calibration = False
        else:
            # New user (or user without saved calibration) — start from scratch.
            system.reset_calibration()
        # Either way, clear any in-progress text from the previous user.
        system.clear_text()

    active_user_id = user_id
    return {"user_id": user_id, "username": username, "has_calibration": has_calibration}


@app.get("/api/calibration")
async def get_calibration(request: Request):
    uid = _require_user_id(request)
    with db_lock:
        row = db_conn.execute(
            "SELECT data_json, updated_at FROM calibrations WHERE user_id = ?",
            (uid,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no calibration on file")
    return {"data": json.loads(row["data_json"]), "updated_at": row["updated_at"]}


@app.post("/api/calibration/load")
async def load_saved_calibration(request: Request):
    """Re-apply the auto-saved calibration for the current user (e.g. after a manual reset)."""
    uid = _require_user_id(request)
    with db_lock:
        row = db_conn.execute(
            "SELECT data_json FROM calibrations WHERE user_id = ?",
            (uid,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no calibration on file")
    cal = deserialize_calibration(row["data_json"])
    with system_lock:
        system.apply_calibration(cal)
    return {"status": "loaded"}


@app.post("/api/calibration/ear/{mode}")
async def collect_ear_baseline(mode: str):
    """Collect EAR baseline for 'open' or 'closed' eye state (~3 s at 15 fps)."""
    global _ear_collect_mode, _ear_samples, _ear_target_count
    if mode not in ('open', 'closed'):
        raise HTTPException(status_code=400, detail="mode must be 'open' or 'closed'")
    if not is_camera_running:
        raise HTTPException(status_code=400, detail="start the camera first")

    _ear_done_event.clear()
    with _ear_collect_lock:
        _ear_samples = []
        _ear_target_count = 45
        _ear_collect_mode = mode

    loop = asyncio.get_event_loop()
    done = await loop.run_in_executor(None, lambda: _ear_done_event.wait(10.0))

    if not done:
        with _ear_collect_lock:
            _ear_collect_mode = None
        raise HTTPException(status_code=408, detail="EAR collection timed out — keep your face in frame")

    with _ear_collect_lock:
        samples = list(_ear_samples)

    if not samples:
        raise HTTPException(status_code=500, detail="no EAR samples collected")

    avg = float(np.mean(samples))
    if mode == 'open':
        system.config.ear_max = avg
        system.calibration_manager.calibration.ear_baseline_open = avg
    else:
        system.config.ear_min = avg
        system.calibration_manager.calibration.ear_baseline_closed = avg

    return {"mode": mode, "avg_ear": round(avg, 4), "samples": len(samples)}


# --- CALIBRATION PROFILE ENDPOINTS ---

@app.get("/api/calibrations")
async def list_calibration_profiles(request: Request):
    uid = _require_user_id(request)
    with db_lock:
        rows = db_conn.execute(
            "SELECT id, name, data_json, created_at FROM calibration_profiles WHERE user_id = ? ORDER BY created_at DESC",
            (uid,),
        ).fetchall()
    profiles = []
    for r in rows:
        d = json.loads(r["data_json"])
        profiles.append({
            "id": r["id"],
            "name": r["name"],
            "created_at": r["created_at"],
            "dot_ms": d.get("avg_dot_duration_ms", 0),
            "dash_ms": d.get("avg_dash_duration_ms", 0),
            "threshold_ms": d.get("avg_blink_duration_ms", 0),
        })
    return {"profiles": profiles}


@app.post("/api/calibrations")
async def save_calibration_profile(request: Request):
    """Save the currently active calibration as a named profile."""
    uid = _require_user_id(request)
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    cal = system.calibration_manager.get_calibration()
    if not cal.is_calibrated:
        raise HTTPException(status_code=400, detail="no active calibration to save")
    payload = serialize_calibration(cal)
    with db_lock:
        cur = db_conn.execute(
            "INSERT INTO calibration_profiles (user_id, name, data_json) VALUES (?, ?, ?)",
            (uid, name, payload),
        )
        db_conn.commit()
        profile_id = cur.lastrowid
    d = json.loads(payload)
    return {
        "id": profile_id,
        "name": name,
        "dot_ms": d["avg_dot_duration_ms"],
        "dash_ms": d["avg_dash_duration_ms"],
        "threshold_ms": d["avg_blink_duration_ms"],
    }


@app.post("/api/calibrations/{profile_id}/load")
async def load_calibration_profile(profile_id: int, request: Request):
    """Load a named calibration profile into the active system."""
    uid = _require_user_id(request)
    with db_lock:
        row = db_conn.execute(
            "SELECT data_json, name FROM calibration_profiles WHERE id = ? AND user_id = ?",
            (profile_id, uid),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="profile not found")
    cal = deserialize_calibration(row["data_json"])
    with system_lock:
        system.apply_calibration(cal)
    return {"status": "loaded", "name": row["name"]}


@app.delete("/api/calibrations/{profile_id}")
async def delete_calibration_profile(profile_id: int, request: Request):
    uid = _require_user_id(request)
    with db_lock:
        result = db_conn.execute(
            "DELETE FROM calibration_profiles WHERE id = ? AND user_id = ?",
            (profile_id, uid),
        )
        db_conn.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="profile not found")
    return {"status": "deleted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)