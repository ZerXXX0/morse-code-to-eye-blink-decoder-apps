import { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

// Mirror of backend implementation.py MORSE_CODE_DICT, inverted for display
// (character -> code). Grouped for readable rendering.
const MORSE_REFERENCE = {
  Letters: [
    ['A', '.-'], ['B', '-...'], ['C', '-.-.'], ['D', '-..'], ['E', '.'],
    ['F', '..-.'], ['G', '--.'], ['H', '....'], ['I', '..'], ['J', '.---'],
    ['K', '-.-'], ['L', '.-..'], ['M', '--'], ['N', '-.'], ['O', '---'],
    ['P', '.--.'], ['Q', '--.-'], ['R', '.-.'], ['S', '...'], ['T', '-'],
    ['U', '..-'], ['V', '...-'], ['W', '.--'], ['X', '-..-'], ['Y', '-.--'],
    ['Z', '--..'],
  ],
  Numbers: [
    ['0', '-----'], ['1', '.----'], ['2', '..---'], ['3', '...--'], ['4', '....-'],
    ['5', '.....'], ['6', '-....'], ['7', '--...'], ['8', '---..'], ['9', '----.'],
  ],
  Punctuation: [
    ['.', '.-.-.-'], [',', '--..--'], ['?', '..--..'], ["'", '.----.'],
    ['!', '-.-.--'], ['/', '-..-.'], ['(', '-.--.'], [')', '-.--.-'],
    ['&', '.-...'], [':', '---...'], [';', '-.-.-.'], ['=', '-...-'],
    ['+', '.-.-.'], ['-', '-....-'], ['_', '..--.-'], ['"', '.-..-.'],
    ['$', '...-..-'], ['@', '.--.-.'],
  ],
  Special: [
    ['SOS', '...---...'],
  ],
};

// Centralized fetch wrapper that injects the X-User-Id header for every API
// call. Auth is explicitly non-secure — the backend just uses the header to
// scope per-user state (mainly calibration persistence).
function apiFetch(path, opts = {}) {
  const userId = localStorage.getItem('userId');
  const headers = { ...(opts.headers || {}) };
  if (userId) headers['X-User-Id'] = userId;
  return fetch(`${API_BASE}${path}`, { ...opts, headers });
}


// --- LOGIN / SIGNUP SCREEN ---
function Login({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!username.trim() || !password) {
      setError('Username and password required');
      return;
    }
    setBusy(true);
    try {
      const path = mode === 'login' ? '/api/auth/login' : '/api/auth/signup';
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || `${mode === 'login' ? 'Sign in' : 'Sign up'} failed`);
        return;
      }
      onLogin({
        id: data.user_id,
        username: data.username,
        hasCalibration: !!data.has_calibration,
      });
    } catch {
      setError('Cannot reach server. Is the backend running?');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-slate-50 font-sans">
      <form onSubmit={submit} className="w-full max-w-sm bg-white border border-slate-200 rounded-2xl shadow-md p-8 space-y-5">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-teal-400 to-cyan-500 rounded-xl shadow-md shadow-teal-500/20 text-white text-3xl mb-3">
            👁️
          </div>
          <h1 className="text-2xl font-extrabold text-slate-800">BlinkLink</h1>
          <p className="text-xs text-teal-600 tracking-widest uppercase font-bold mt-1">
            {mode === 'login' ? 'Sign in to continue' : 'Create your account'}
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-600 mb-1">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
            className="w-full p-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-teal-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-600 mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-teal-500 focus:outline-none"
          />
        </div>

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-700 text-sm rounded-lg p-3">{error}</div>
        )}

        <button
          type="submit"
          disabled={busy}
          className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 text-white p-2 rounded-lg font-medium shadow-sm transition-colors"
        >
          {busy ? '…' : mode === 'login' ? 'Sign In' : 'Sign Up'}
        </button>

        <p className="text-center text-sm text-slate-500">
          {mode === 'login' ? (
            <>No account?{' '}
              <button type="button" onClick={() => { setMode('signup'); setError(''); }} className="text-teal-700 font-medium hover:underline">
                Sign up
              </button>
            </>
          ) : (
            <>Already have one?{' '}
              <button type="button" onClick={() => { setMode('login'); setError(''); }} className="text-teal-700 font-medium hover:underline">
                Sign in
              </button>
            </>
          )}
        </p>

        <p className="text-center text-[11px] text-slate-400 leading-snug">
          Auth here isn't for security — it just keeps your calibration<br />separate from other users on this machine.
        </p>
      </form>
    </div>
  );
}

function App() {
  // --- AUTH STATE ---
  const [currentUser, setCurrentUser] = useState(() => {
    const id = localStorage.getItem('userId');
    const name = localStorage.getItem('username');
    return id && name ? { id: parseInt(id, 10), username: name } : null;
  });

  // --- STATE MANAGEMENT ---
  const [letterGap, setLetterGap] = useState(1.5);
  const [wordGap, setWordGap] = useState(3.0);
  const [sentenceGap, setSentenceGap] = useState(5.0);
  const [nlpEnabled, setNlpEnabled] = useState(false);
  const [showMorseRef, setShowMorseRef] = useState(false);
  const [focusMode, setFocusMode] = useState(false);

  const [eyeState, setEyeState] = useState("UNKNOWN");
  const [confidence, setConfidence] = useState(0.0);
  const [fps, setFps] = useState(0.0);
  const [currentMorse, setCurrentMorse] = useState("");

  const [decodedText, setDecodedText] = useState("");
  const [nlpText, setNlpText] = useState("");
  const [calStatus, setCalStatus] = useState("Connecting to Engine...");

  // Pending slider updates — batched + debounced into a single POST. Each
  // movement merges into pendingConfig; a 200ms window of stillness flushes
  // it. Pure React state keeps the react-hooks lint happy (no refs to
  // mutate, no module-level side effects).
  const [pendingConfig, setPendingConfig] = useState({});

  useEffect(() => {
    if (Object.keys(pendingConfig).length === 0) return;
    const timer = setTimeout(async () => {
      try {
        await apiFetch('/api/update_config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(pendingConfig),
        });
      } catch (e) {
        console.error('Gagal update config', e);
      } finally {
        setPendingConfig({});
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [pendingConfig]);

  // --- INTEGRASI WEBSOCKET (REAL-TIME DATA) ---
  // Only open the WebSocket once a user is logged in — otherwise we'd push
  // engine state into the Login screen where it's unused.
  useEffect(() => {
    if (!currentUser) return;
    const wsUrl = API_BASE.replace(/^http/, 'ws') + '/ws/data';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Terhubung ke peladen AI!');
      setCalStatus((prev) => prev.startsWith('✅') ? prev : 'System Online - Connected to Engine');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.eyeState) setEyeState(data.eyeState);
        if (data.confidence !== undefined) setConfidence(data.confidence);
        if (data.fps !== undefined) setFps(data.fps);
        if (data.morseSequence !== undefined) setCurrentMorse(data.morseSequence);
        if (data.decodedText !== undefined) setDecodedText(data.decodedText);
        if (data.nlpText !== undefined) setNlpText(data.nlpText);

        if (data.isCalibrating) {
          const [phase, current, target] = data.calProgress;
          setCalStatus(`🎯 Calibrating ${phase}: ${current} / ${target} blinks`);
        } else if (data.isCalibrating === false) {
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
    };

    return () => {
      ws.close();
    };
  }, [currentUser]);

  // --- ESC EXITS FOCUS MODE ---
  useEffect(() => {
    if (!focusMode) return;
    const onKey = (e) => {
      if (e.key === 'Escape') setFocusMode(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focusMode]);

  // --- AUTH HANDLERS ---
  const handleLogin = (user) => {
    localStorage.setItem('userId', String(user.id));
    localStorage.setItem('username', user.username);
    setCurrentUser({ id: user.id, username: user.username });
    setCalStatus(
      user.hasCalibration
        ? '✅ Calibration loaded from your account. Ready to type.'
        : 'Welcome! Run Begin Cal. to set your blink thresholds.'
    );
  };

  const handleSignOut = () => {
    localStorage.removeItem('userId');
    localStorage.removeItem('username');
    setCurrentUser(null);
    // Drop transient UI state so a fresh signin doesn't see stale data.
    setDecodedText('');
    setNlpText('');
    setCurrentMorse('');
    setEyeState('UNKNOWN');
    setCalStatus('Connecting to Engine...');
  };

  // --- INTEGRASI REST API (KONTROL TOMBOL & INPUT) ---

  const handleSliderChange = (key, setter) => (e) => {
    const val = parseFloat(e.target.value);
    setter(val);
    setPendingConfig(prev => ({ ...prev, [key]: val }));
  };

  const handleNlpToggle = async () => {
    try {
      const res = await apiFetch('/api/toggle_nlp', { method: 'POST' });
      const data = await res.json();
      if (typeof data.enabled === 'boolean') {
        setNlpEnabled(data.enabled);
      } else {
        setNlpEnabled((prev) => !prev);
      }
    } catch (e) {
      console.error("Gagal toggle NLP", e);
    }
  };

  const handleStartCalibration = async () => {
    setCalStatus("⏳ Starting Calibration...");
    try {
      await apiFetch('/api/start_calibration', { method: 'POST' });
    } catch (e) {
      console.error("Gagal memulai kalibrasi", e);
    }
  };

  const handleNextStep = async () => {
    try {
      await apiFetch('/api/next_step', { method: 'POST' });
    } catch (e) {
      console.error("Gagal lanjut ke tahap kalibrasi berikutnya", e);
    }
  };

  const handleResetCalibration = async () => {
    setCalStatus("🔄 Resetting calibration...");
    try {
      await apiFetch('/api/reset_calibration', { method: 'POST' });
      setCalStatus("ℹ️ Calibration Reset to Default. Ready.");
    } catch (e) {
      console.error("Gagal mereset kalibrasi", e);
      setCalStatus("❌ Error resetting calibration!");
    }
  };

  const handleClearText = async () => {
    setDecodedText("");
    setNlpText("");
    try {
      await apiFetch('/api/clear_text', { method: 'POST' });
    } catch (error) {
      console.error("Gagal mengirim perintah reset teks ke peladen:", error);
    }
  };

  const handleCameraStart = async () => {
    try {
      await apiFetch('/api/camera/start', { method: 'POST' });
      setCalStatus("📷 Camera starting...");
    } catch (e) { console.error("Gagal start kamera"); }
  };

  const handleCameraStop = async () => {
    try {
      await apiFetch('/api/camera/stop', { method: 'POST' });
      setCalStatus("📷 Camera stopped.");
    } catch (e) { console.error("Gagal stop kamera"); }
  };

  const handleCameraReset = async () => {
    try {
      await apiFetch('/api/camera/reset', { method: 'POST' });
      setDecodedText("");
      setNlpText("");
      setCalStatus("🔄 System & Camera Reset.");
    } catch (e) { console.error("Gagal reset kamera"); }
  };

  const goFullscreen = () => {
    const el = document.documentElement;
    if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
  };

  // --- AUTH GATE ---
  if (!currentUser) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <>
      <div className="flex h-screen bg-slate-50 text-slate-800 font-sans">

        {/* SIDEBAR */}
        <div className="w-80 bg-white p-6 overflow-y-auto border-r border-slate-200 flex flex-col shadow-sm z-10">
          <h2 className="text-xl font-bold mb-6 text-slate-800">⚙️ Settings</h2>

          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 mb-2">Letter Gap: {letterGap.toFixed(2)}s</label>
            <input
              type="range" min="0.3" max="5" step="0.1" value={letterGap}
              onChange={handleSliderChange('letter_gap_seconds', setLetterGap)}
              className="w-full accent-teal-600"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 mb-2">Word Gap: {wordGap.toFixed(2)}s</label>
            <input
              type="range" min="0.5" max="8" step="0.1" value={wordGap}
              onChange={handleSliderChange('word_gap_seconds', setWordGap)}
              className="w-full accent-teal-600"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 mb-2">Sentence Gap: {sentenceGap.toFixed(2)}s</label>
            <input
              type="range" min="1" max="12" step="0.1" value={sentenceGap}
              onChange={handleSliderChange('sentence_gap_seconds', setSentenceGap)}
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
                <h1 className="text-3xl font-extrabold text-slate-800">BlinkLink</h1>
                <p className="text-xs text-teal-600 tracking-widest uppercase font-bold">Assistive Communication</p>
              </div>
            </div>

            <div className="flex items-center gap-3 text-sm">
              <span className="px-3 py-1 bg-slate-100 text-slate-700 border border-slate-200 rounded-full font-medium">
                Signed in as <span className="font-mono">{currentUser.username}</span>
              </span>
              <button
                onClick={handleSignOut}
                className="px-3 py-1 bg-white border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-full font-medium transition-colors"
              >
                Sign out
              </button>
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
                  <button
                    onClick={() => setFocusMode(true)}
                    className="ml-auto text-xs bg-slate-100 hover:bg-slate-200 border border-slate-200 px-3 py-1 rounded-full font-medium text-slate-700 transition-colors"
                    title="Expand video to fill the window"
                  >
                    ⛶ Focus Mode
                  </button>
                </h3>

                <div className="w-full h-64 bg-slate-900 rounded-xl flex items-center justify-center mb-5 border border-slate-200 overflow-hidden relative shadow-inner">
                  <div className="absolute inset-0 border-2 border-teal-500/30 rounded-xl m-4 pointer-events-none z-10"></div>

                  <img
                    src={`${API_BASE}/video_feed`}
                    alt="Webcam Stream"
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <button onClick={handleCameraStart} className="bg-teal-600 hover:bg-teal-700 text-white transition-colors p-2 rounded-lg font-medium shadow-sm">▶ Start</button>
                  <button onClick={handleCameraStop} className="bg-rose-500 hover:bg-rose-600 text-white transition-colors p-2 rounded-lg font-medium shadow-sm">⏹ Stop</button>
                  <button onClick={handleCameraReset} className="bg-slate-100 hover:bg-slate-200 border border-slate-300 text-slate-700 transition-colors p-2 rounded-lg font-medium">🔄 Reset</button>
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

            {/* MORSE REFERENCE REVEAL */}
            <div className="mt-4">
              <button
                onClick={() => setShowMorseRef((v) => !v)}
                aria-expanded={showMorseRef}
                className="w-full bg-white border border-slate-200 hover:bg-slate-50 transition-colors p-3 rounded-xl shadow-sm flex items-center justify-between text-slate-700 font-medium"
              >
                <span className="flex items-center gap-2">
                  <span className="text-amber-600">·−</span>
                  Morse Code Reference
                </span>
                <span className={`transform transition-transform ${showMorseRef ? 'rotate-180' : ''}`}>▾</span>
              </button>

              {showMorseRef && (
                <div className="mt-3 bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-5">
                  {Object.entries(MORSE_REFERENCE).map(([group, items]) => (
                    <div key={group}>
                      <h4 className="text-xs uppercase tracking-wider font-bold text-slate-500 mb-2">{group}</h4>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                        {items.map(([char, code]) => (
                          <div
                            key={char}
                            className="flex items-center justify-between bg-slate-50 border border-slate-100 rounded-lg px-3 py-2"
                          >
                            <span className="font-mono font-bold text-slate-800">{char}</span>
                            <span className="font-mono text-amber-600 tracking-widest">{code}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>
        </div>
      </div>

      {/* FOCUS MODE OVERLAY */}
      {/* Rendered alongside (not instead of) the dashboard so the dashboard
          <img> stays mounted — preventing an MJPEG reconnect on toggle. The
          overlay uses a separate <img> hitting the same stream. */}
      {focusMode && (
        <div className="fixed inset-0 z-50 bg-black">
          <img
            src={`${API_BASE}/video_feed`}
            alt="Webcam Stream (Focus)"
            className="absolute inset-0 w-full h-full object-cover"
          />

          <div className="absolute top-4 right-4 z-10 flex gap-2">
            <button
              onClick={goFullscreen}
              className="bg-white/10 hover:bg-white/20 backdrop-blur text-white text-sm px-3 py-2 rounded-lg border border-white/20 transition-colors"
            >
              ⛶ Go Fullscreen
            </button>
            <button
              onClick={() => setFocusMode(false)}
              className="bg-white/10 hover:bg-white/20 backdrop-blur text-white text-sm px-3 py-2 rounded-lg border border-white/20 transition-colors"
            >
              ✕ Exit Focus
            </button>
          </div>

          <div className="absolute bottom-0 inset-x-0 bg-black/70 backdrop-blur-sm text-white py-5 px-6 flex flex-col items-center gap-2 z-10">
            <p className="text-amber-400 tracking-[0.4em] text-3xl font-mono font-bold min-h-[1em]">
              {currentMorse || '·'}
            </p>
            <p className="text-xl font-mono text-center max-w-3xl break-words">
              {decodedText
                ? decodedText.slice(-40)
                : <span className="text-white/40 italic">Waiting for input…</span>}
            </p>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
