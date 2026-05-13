import cv2
import json
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

# Inisialisasi sistem AI
config = SystemConfig()
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
    """Endpoint untuk stream gambar ke tag <img> di React."""
    def generate_frames():
        while True:
            if latest_frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                # Jika kamera mati, kirim frame kosong atau tunggu
                import time
                time.sleep(0.1)
    
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
    system.update_config(alpha=data.get("alpha", 0.4))
    return {"status": "updated"}

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