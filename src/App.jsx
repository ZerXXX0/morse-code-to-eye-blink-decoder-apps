import { useState, useEffect } from 'react';

function App() {
  // --- STATE MANAGEMENT ---
  const [alpha, setAlpha] = useState(0.4);
  const [nlpEnabled, setNlpEnabled] = useState(false);
  
  const [eyeState, setEyeState] = useState("UNKNOWN");
  const [confidence, setConfidence] = useState(0.0);
  const [fps, setFps] = useState(0.0);
  const [currentMorse, setCurrentMorse] = useState("");
  
  const [decodedText, setDecodedText] = useState("");
  const [nlpText, setNlpText] = useState("");
  const [calStatus, setCalStatus] = useState("Connecting to Engine...");

  // --- INTEGRASI WEBSOCKET (REAL-TIME DATA) ---
  useEffect(() => {
    // Membuka koneksi WebSocket ke FastAPI backend
    const ws = new WebSocket('ws://localhost:8000/ws/data');

    ws.onopen = () => {
      console.log('Terhubung ke peladen AI!');
      setCalStatus("System Online - Connected to Engine");
    };

    ws.onmessage = (event) => {
      // Menangkap data JSON dari backend dan memperbarui UI
      try {
        const data = JSON.parse(event.data);
        if (data.eyeState) setEyeState(data.eyeState);
        if (data.confidence !== undefined) setConfidence(data.confidence);
        if (data.fps !== undefined) setFps(data.fps);
        if (data.morseSequence !== undefined) setCurrentMorse(data.morseSequence);
        if (data.decodedText !== undefined) setDecodedText(data.decodedText);
        if (data.nlpText !== undefined) setNlpText(data.nlpText);

        // --- UPDATE: Logika Kalibrasi Live ---
        if (data.isCalibrating) {
          const [phase, current, target] = data.calProgress;
          setCalStatus(`🎯 Calibrating ${phase}: ${current} / ${target} blinks`);
        } else if (data.isCalibrating === false) {
          // Menggunakan 'prev' agar React selalu mengecek tulisan terakhir di layar
          setCalStatus(prev => {
            if (prev.startsWith("🎯 Calibrating")) {
              return "✅ Calibration Complete! Ready to type.";
            }
            return prev;
          });
        }
      } catch (error) {
        console.error("Error parsing WebSocket data:", error);
      }
    };

    ws.onerror = (error) => {
      console.error('Koneksi WebSocket bermasalah:', error);
      setCalStatus("Connection Error! Is backend running?");
    };

    ws.onclose = () => {
      console.log('Terputus dari peladen AI');
      setCalStatus("Disconnected from Engine");
    };

    // Bersihkan koneksi saat komponen/halaman ditutup
    return () => {
      ws.close();
    };
  }, []); 

  // --- INTEGRASI REST API (KONTROL TOMBOL & INPUT) ---

  // 1. Update Alpha (Dikirim saat slider digeser)
  const handleAlphaChange = async (val) => {
    setAlpha(val);
    try {
      await fetch('http://localhost:8000/api/update_config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alpha: parseFloat(val) })
      });
    } catch (e) { 
      console.error("Gagal update Alpha", e); 
    }
  };

  // 2. Toggle NLP (Dikirim saat checkbox diklik)
  const handleNlpToggle = async () => {
    const newVal = !nlpEnabled;
    setNlpEnabled(newVal);
    try {
      await fetch('http://localhost:8000/api/toggle_nlp', { method: 'POST' });
    } catch (e) { 
      console.error("Gagal toggle NLP", e); 
    }
  };

  // 3. Kontrol Kalibrasi
  const handleStartCalibration = async () => {
    setCalStatus("⏳ Starting Calibration...");
    try {
      await fetch('http://localhost:8000/api/start_calibration', { method: 'POST' });
    } catch (e) { 
      console.error("Gagal memulai kalibrasi", e); 
    }
  };

  const handleNextStep = async () => {
    try {
      await fetch('http://localhost:8000/api/next_step', { method: 'POST' });
    } catch (e) { 
      console.error("Gagal lanjut ke tahap kalibrasi berikutnya", e); 
    }
  };

  // --- UPDATE: Logika Reset Kalibrasi ---
  const handleResetCalibration = async () => {
    setCalStatus("🔄 Resetting calibration...");
    try {
      await fetch('http://localhost:8000/api/reset_calibration', { method: 'POST' });
      setCalStatus("ℹ️ Calibration Reset to Default. Ready.");
    } catch (e) { 
      console.error("Gagal mereset kalibrasi", e); 
      setCalStatus("❌ Error resetting calibration!");
    }
  };

  // 4. Kontrol Hapus Teks
  const handleClearText = async () => {
    setDecodedText("");
    setNlpText("");
    try {
      await fetch('http://localhost:8000/api/clear_text', { method: 'POST' });
    } catch (error) {
      console.error("Gagal mengirim perintah reset teks ke peladen:", error);
    }
  };

  // --- KONTROL KAMERA ---
  const handleCameraStart = async () => {
    try {
      await fetch('http://localhost:8000/api/camera/start', { method: 'POST' });
      setCalStatus("📷 Camera starting...");
    } catch (e) { console.error("Gagal start kamera"); }
  };

  const handleCameraStop = async () => {
    try {
      await fetch('http://localhost:8000/api/camera/stop', { method: 'POST' });
      setCalStatus("📷 Camera stopped.");
    } catch (e) { console.error("Gagal stop kamera"); }
  };

  const handleCameraReset = async () => {
    try {
      await fetch('http://localhost:8000/api/camera/reset', { method: 'POST' });
      setDecodedText("");
      setNlpText("");
      setCalStatus("🔄 System & Camera Reset.");
    } catch (e) { console.error("Gagal reset kamera"); }
  };

  return (
    // TEMA CERAH (Soft Light)
    <div className="flex h-screen bg-slate-50 text-slate-800 font-sans">
      
      {/* SIDEBAR */}
      <div className="w-80 bg-white p-6 overflow-y-auto border-r border-slate-200 flex flex-col shadow-sm z-10">
        <h2 className="text-xl font-bold mb-6 text-slate-800">⚙️ Settings</h2>
        
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-600 mb-2">Alpha (YOLO weight): {alpha}</label>
          <input 
            type="range" min="0" max="1" step="0.05" value={alpha} 
            onChange={(e) => handleAlphaChange(e.target.value)}
            className="w-full accent-teal-600"
          />
        </div>

        <div className="mb-6">
          <label className="flex items-center space-x-3 cursor-pointer">
            <input 
              type="checkbox" checked={nlpEnabled} 
              onChange={handleNlpToggle}
              className="w-4 h-4 accent-teal-600 rounded bg-slate-100 border-slate-300"
            />
            <span className="text-slate-700 font-medium">Enable NLP Correction</span>
          </label>
        </div>

        <hr className="border-slate-100 my-4" />

        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2 text-slate-800">🎯 Calibration</h3>
          <p className="text-xs text-slate-500 mb-4">Required: EAR open/closed, then dot/dash</p>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <button onClick={handleStartCalibration} className="bg-teal-600 hover:bg-teal-700 text-white transition-colors p-2 rounded text-sm font-medium shadow-sm">Begin Cal.</button>
            <button onClick={handleNextStep} className="bg-slate-200 hover:bg-slate-300 text-slate-700 transition-colors p-2 rounded text-sm font-medium">Next Step</button>
          </div>
          <button onClick={handleResetCalibration} className="w-full bg-rose-500 hover:bg-rose-600 text-white transition-colors p-2 rounded text-sm font-medium shadow-sm">Reset Cal.</button>
        </div>

        <hr className="border-slate-100 my-4" />

        <div className="mt-auto">
          <h3 className="text-lg font-semibold mb-2 text-slate-800">📝 Text Controls</h3>
          <button onClick={handleClearText} className="w-full bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 transition-colors p-2 rounded text-sm font-medium mb-2 shadow-sm">Clear Text</button>
        </div>
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col overflow-hidden">
        
        {/* HEADER */}
        <header className="flex items-center justify-between p-6 bg-white/80 backdrop-blur-md border-b border-slate-200 z-10">
          <div className="flex items-center gap-4">
            <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-teal-400 to-cyan-500 rounded-xl shadow-md shadow-teal-500/20 text-white">
              <span className="text-2xl">👁️</span>
            </div>
            <div>
              <h1 className="text-3xl font-extrabold text-slate-800">
                BlinkLink
              </h1>
              <p className="text-xs text-teal-600 tracking-widest uppercase font-bold">Assistive Communication</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 text-sm">
            <span className="px-3 py-1 bg-teal-50 text-teal-700 border border-teal-200 rounded-full font-mono font-medium shadow-sm">
              System Online
            </span>
            <span className="px-3 py-1 bg-slate-100 text-slate-600 border border-slate-200 rounded-full font-medium">
              v1.0-beta
            </span>
          </div>
        </header>

        {/* SCROLLABLE DASHBOARD CONTENT */}
        <div className="p-8 overflow-y-auto">
          
          <div className="grid grid-cols-12 gap-6 mb-8">
            
            {/* KOLOM 1: Live Video */}
            <div className="col-span-5 bg-white border border-slate-200 p-5 rounded-2xl shadow-sm">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-800">
                <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
                Live Video
              </h3>
              
              {/* INTEGRASI VIDEO STREAM */}
              <div className="w-full h-64 bg-slate-900 rounded-xl flex items-center justify-center mb-5 border border-slate-200 overflow-hidden relative shadow-inner">
                <div className="absolute inset-0 border-2 border-teal-500/30 rounded-xl m-4 pointer-events-none z-10"></div>
                
                <img 
                  src="http://localhost:8000/video_feed" 
                  alt="Webcam Stream" 
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    if (e.target.nextSibling) e.target.nextSibling.style.display = 'block';
                  }}
                />
                <span className="text-slate-400 font-medium hidden">Gagal memuat video... (Cek Backend)</span>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <button 
                  onClick={handleCameraStart} 
                  className="bg-teal-600 hover:bg-teal-700 text-white transition-colors p-2 rounded-lg font-medium shadow-sm"
                >▶ Start</button>
  
                <button 
                  onClick={handleCameraStop} 
                  className="bg-rose-500 hover:bg-rose-600 text-white transition-colors p-2 rounded-lg font-medium shadow-sm"
                >⏹ Stop</button>
  
                <button 
                  onClick={handleCameraReset} 
                  className="bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 transition-colors p-2 rounded-lg font-medium"
                >🔄 Reset</button>
              </div>
            </div>

            {/* KOLOM 2: Status */}
            <div className="col-span-3 bg-white border border-slate-200 p-5 rounded-2xl shadow-sm flex flex-col justify-between">
              <div>
                <h3 className="text-lg font-bold mb-5 text-slate-800">📊 Status</h3>
                <div className="mb-4 bg-slate-50 p-4 rounded-xl border border-slate-100">
                  <p className="text-xs text-slate-500 uppercase tracking-wider mb-1 font-semibold">Eye State</p>
                  <p className="text-2xl font-mono text-teal-700 font-bold">{eyeState === "OPEN" ? "👁️ " : "😑 "} {eyeState}</p>
                </div>
                <div className="mb-6 flex gap-4">
                  <div className="flex-1 bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1 font-semibold">Conf</p>
                    <p className="text-xl font-mono font-bold text-slate-700">{(confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div className="flex-1 bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1 font-semibold">FPS</p>
                    <p className="text-xl font-mono font-bold text-slate-700">{fps}</p>
                  </div>
                </div>
              </div>
              <div className="bg-amber-50 p-4 rounded-xl border border-amber-100">
                <p className="text-xs text-amber-700 uppercase tracking-wider mb-1 font-bold">Current Morse</p>
                <p className="text-3xl font-bold text-amber-600 tracking-[0.3em]">{currentMorse || "..."}</p>
              </div>
            </div>

            {/* KOLOM 3: Text Output */}
            <div className="col-span-4 space-y-4 flex flex-col">
              <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm flex-1 flex flex-col">
                <h3 className="text-sm font-bold text-slate-600 uppercase tracking-wider mb-3 flex justify-between items-center">
                  Raw Decoded
                  <span className="text-xs bg-slate-100 px-2 py-1 rounded text-slate-600 border border-slate-200">CER: 3.6%</span>
                </h3>
                <p className="text-xl font-mono text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-xl flex-1 border border-slate-100 shadow-inner">
                  {decodedText || <span className="text-slate-400 italic">Waiting for input...</span>}
                </p>
              </div>
              <div className="bg-gradient-to-br from-teal-50 to-cyan-50 border border-teal-100 p-5 rounded-2xl shadow-sm flex-1 flex flex-col relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-teal-200/30 rounded-full blur-3xl"></div>
                <h3 className="text-sm font-bold text-teal-800 uppercase tracking-wider mb-3 flex justify-between items-center relative z-10">
                  NLP Corrected
                  <span className="text-xs bg-white/60 px-2 py-1 rounded text-teal-700 border border-teal-200 shadow-sm">IndoBERT</span>
                </h3>
                <p className="text-2xl font-semibold text-teal-900 leading-relaxed relative z-10">
                  {nlpText || <span className="text-teal-600/50 italic">No output yet</span>}
                </p>
              </div>
            </div>

          </div>

          {/* BOTTOM ROW: Calibration Status */}
          <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm flex items-center gap-3">
            <span className="flex h-3 w-3 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
            </span>
            <p className="text-slate-600 font-mono text-sm tracking-wide font-medium">{calStatus}</p>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;