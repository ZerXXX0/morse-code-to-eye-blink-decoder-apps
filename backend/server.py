import cv2
import json
import asyncio
import os
import sys
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Resolve paths relative to this file so the server works regardless of CWD
# (e.g. `python backend/server.py` from the project root).
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Import sistem utama dari file implementation.py
from implementation import EyeBlinkMorseSystem, SystemConfig

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

# Variabel Global
latest_frame = None
latest_data = {}
is_camera_running = False
camera_thread = None

def camera_loop():
    """Background thread untuk menangkap kamera dan menjalankan AI."""
    global latest_frame, latest_data, is_camera_running
    
    cap = cv2.VideoCapture(0)
    # Gunakan resolusi standar agar tidak terlalu berat
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("LOG: Camera Hardware Opened")
    
    while is_camera_running:
        ret, frame = cap.read()
        if not ret:
            print("LOG: Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1) # Mirror effect
        
        # Proses frame ke AI
        annotated_frame, results = system.process_frame(frame, enable_detection=True)
        
        # Simpan metrik untuk dikirim via WebSocket
        latest_data = {
            "eyeState": results['eye_state'].value.upper(),
            "confidence": results['confidence'],
            "fps": round(system.blink_detector.estimated_fps, 1),
            "morseSequence": results['morse_sequence'],
            "decodedText": results['decoded_text'],
            "nlpText": results.get('nlp_smoothed_text', ''),
            "isCalibrating": results.get('is_calibrating', False),
            "calProgress": results.get('calibration_progress', ("DONE", 0, 0))
        }
        
        # Encode gambar ke JPG untuk dikirim via MJPEG Stream
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        latest_frame = buffer.tobytes()

    # Tutup kamera saat is_camera_running jadi False
    cap.release()
    latest_frame = None
    print("LOG: Camera Hardware Released")

# --- ENDPOINT KONTROL KAMERA ---

@app.post("/api/camera/start")
async def start_camera_api():
    global is_camera_running, camera_thread
    if not is_camera_running:
        is_camera_running = True
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        return {"status": "camera started"}
    return {"status": "camera already running"}

@app.post("/api/camera/stop")
async def stop_camera_api():
    global is_camera_running, latest_frame
    is_camera_running = False
    latest_frame = None 
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
async def start_calibration():
    system.start_calibration()
    return {"status": "started"}

@app.post("/api/next_step")
async def next_step():
    if hasattr(system, 'next_calibration_step'):
        system.next_calibration_step()
    return {"status": "next_step_triggered"}

@app.post("/api/reset_calibration")
async def reset_calibration():
    system.reset_calibration()
    return {"status": "reset"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)