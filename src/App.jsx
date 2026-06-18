import { useState, useEffect } from 'react';
import blinklinkLogo from './assets/blinklink_logo.png';

const API_BASE = 'http://localhost:8000';

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
  Special: [['SOS', '...---...']],
};

function apiFetch(path, opts = {}) {
  const userId = localStorage.getItem('userId');
  const headers = { ...(opts.headers || {}) };
  if (userId) headers['X-User-Id'] = userId;
  return fetch(`${API_BASE}${path}`, { ...opts, headers });
}

// ─────────────────────────────────────────────────────────────────────────────
// LOGIN SCREEN
// ─────────────────────────────────────────────────────────────────────────────
function Login({ onLogin }) {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    if (!username.trim() || !password) { setError('Username and password required'); return; }
    setBusy(true);
    try {
      const path = mode === 'login' ? '/api/auth/login' : '/api/auth/signup';
      const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setError(data.detail || `${mode === 'login' ? 'Sign in' : 'Sign up'} failed`); return; }
      onLogin({ id: data.user_id, username: data.username, hasCalibration: !!data.has_calibration });
    } catch {
      setError('Cannot reach server. Is the backend running?');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-slate-50 dark:bg-slate-950 font-sans transition-colors duration-300">
      <form onSubmit={submit} className="w-full max-w-sm bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl shadow-md p-8 space-y-5">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-teal-400 to-cyan-500 rounded-xl shadow-md shadow-teal-500/20 text-white text-3xl mb-3">👁️</div>
          <h1 className="text-2xl font-extrabold text-slate-800 dark:text-slate-100">BlinkLink</h1>
          <p className="text-xs text-teal-600 dark:text-teal-400 tracking-widest uppercase font-bold mt-1">
            {mode === 'login' ? 'Sign in to continue' : 'Create your account'}
          </p>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">Username</label>
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} autoFocus
            className="w-full p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-800 dark:text-slate-100 focus:ring-2 focus:ring-teal-500 focus:outline-none" />
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-1">Password</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-800 dark:text-slate-100 focus:ring-2 focus:ring-teal-500 focus:outline-none" />
        </div>
        {error && <div className="bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/40 text-rose-700 dark:text-rose-400 text-sm rounded-lg p-3">{error}</div>}
        <button type="submit" disabled={busy}
          className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white p-2 rounded-lg font-medium shadow-sm transition-colors">
          {busy ? '…' : mode === 'login' ? 'Sign In' : 'Sign Up'}
        </button>
        <p className="text-center text-sm text-slate-500 dark:text-slate-400">
          {mode === 'login' ? (
            <>No account?{' '}<button type="button" onClick={() => { setMode('signup'); setError(''); }} className="text-teal-700 dark:text-teal-400 font-medium hover:underline">Sign up</button></>
          ) : (
            <>Already have one?{' '}<button type="button" onClick={() => { setMode('login'); setError(''); }} className="text-teal-700 dark:text-teal-400 font-medium hover:underline">Sign in</button></>
          )}
        </p>
        <p className="text-center text-[11px] text-slate-400 dark:text-slate-500 leading-snug">
          Auth here isn't for security — it just keeps your calibration<br />separate from other users on this machine.
        </p>
      </form>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CALIBRATION MODAL
// ─────────────────────────────────────────────────────────────────────────────
function CalibrationModal({ onClose, onStatusChange, onCameraStart, isCameraRunning, isCalibrating, calProgress, calData, currentEar }) {
  const [step, setStep] = useState(0);
  const [blinkCount, setBlinkCount] = useState(3);

  // Auto-start the camera when the calibration modal opens.
  useEffect(() => {
    if (!isCameraRunning) onCameraStart();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const [earOpenResult, setEarOpenResult] = useState(null);
  const [earClosedResult, setEarClosedResult] = useState(null);
  const [capturing, setCapturing] = useState(false);
  const [captureError, setCaptureError] = useState('');
  const [profileName, setProfileName] = useState('');
  const [savingProfile, setSavingProfile] = useState(false);

  const STEP_LABELS = ['Setup', 'Eye Open', 'Eye Closed', 'Dot Blinks', 'Dash Blinks', 'Done'];

  // Auto-advance from dots to dashes
  useEffect(() => {
    if (step === 3 && isCalibrating && calProgress[0] === 'DASHES') setStep(4);
  }, [isCalibrating, calProgress, step]);

  // Auto-advance to complete when calibration finishes
  useEffect(() => {
    if (step >= 3 && step < 5 && !isCalibrating && calData.isCalibrated) {
      setStep(5);
      onStatusChange('✅ Calibration Complete! Ready to type.');
    }
  }, [isCalibrating, calData.isCalibrated, step]);

  const handleEarCapture = async (mode) => {
    setCaptureError('');
    setCapturing(true);
    try {
      const res = await apiFetch(`/api/calibration/ear/${mode}`, { method: 'POST' });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { setCaptureError(data.detail || 'Capture failed'); return; }
      if (mode === 'open') setEarOpenResult(data.avg_ear);
      else setEarClosedResult(data.avg_ear);
    } catch {
      setCaptureError('Failed to reach server');
    } finally {
      setCapturing(false);
    }
  };

  const handleStartBlinkCal = async () => {
    setCaptureError('');
    try {
      await apiFetch('/api/start_calibration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ blink_count: blinkCount }),
      });
      setStep(3);
    } catch {
      setCaptureError('Failed to start calibration');
    }
  };

  const handleNextStep = async () => {
    await apiFetch('/api/next_step', { method: 'POST' });
  };

  const handleSaveProfile = async () => {
    if (!profileName.trim()) return;
    setSavingProfile(true);
    try {
      const res = await apiFetch('/api/calibrations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: profileName.trim() }),
      });
      if (res.ok) {
        onStatusChange(`✅ Profile "${profileName.trim()}" saved.`);
        onClose();
      } else {
        const d = await res.json().catch(() => ({}));
        setCaptureError(d.detail || 'Failed to save profile');
      }
    } catch {
      setCaptureError('Failed to save profile');
    } finally {
      setSavingProfile(false);
    }
  };

  const dotCurrent = calProgress[0] === 'DOTS' ? (calProgress[1] ?? 0) : 0;
  const dashCurrent = calProgress[0] === 'DASHES' ? (calProgress[1] ?? 0) : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden border border-slate-200 dark:border-slate-800">

        {/* Header with step indicator */}
        <div className="bg-gradient-to-r from-teal-500 to-cyan-500 p-5 text-white">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold">🎯 Calibration</h2>
            <button onClick={onClose} className="text-white/70 hover:text-white text-2xl leading-none font-bold">×</button>
          </div>
          <div className="flex gap-1">
            {STEP_LABELS.map((label, i) => (
              <div key={i} className={`flex-1 text-center text-[9px] font-semibold py-1 rounded transition-colors ${
                i === step ? 'bg-white text-teal-700' :
                i < step  ? 'bg-white/40 text-white' : 'bg-white/15 text-white/50'
              }`}>{label}</div>
            ))}
          </div>
        </div>

        <div className="p-6">

          {/* Camera warning */}
          {!isCameraRunning && step > 0 && step < 5 && (
            <div className="mb-4 bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-900/40 text-amber-800 dark:text-amber-300 text-sm rounded-lg p-3">
              ⚠️ Camera must be running for calibration. Start it from the main screen.
            </div>
          )}

          {/* Error */}
          {captureError && (
            <div className="mb-4 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-900/40 text-rose-700 dark:text-rose-400 text-sm rounded-lg p-3">{captureError}</div>
          )}

          {/* ── Step 0: Setup ───────────────────────────────────────── */}
          {step === 0 && (
            <div className="space-y-5">
              <div>
                <p className="text-slate-800 dark:text-slate-100 font-semibold mb-1">Blinks per type</p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
                  You'll do this many short blinks, then this many long blinks.
                </p>
                <div className="flex gap-2">
                  {[3, 5, 7, 10].map(n => (
                    <button key={n} onClick={() => setBlinkCount(n)}
                      className={`flex-1 py-2.5 rounded-xl font-mono font-bold text-sm border-2 transition-colors ${
                        blinkCount === n
                          ? 'bg-teal-600 text-white border-teal-600'
                          : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-teal-400 dark:hover:border-teal-500'
                      }`}>
                      {n}
                    </button>
                  ))}
                </div>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Steps: measure open-eye EAR → closed-eye EAR → {blinkCount} short blinks → {blinkCount} long blinks.
              </p>
              <button onClick={() => { setCaptureError(''); setStep(1); }} disabled={!isCameraRunning}
                className="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white p-3 rounded-xl font-semibold transition-colors">
                {isCameraRunning ? 'Begin →' : 'Start camera first'}
              </button>
            </div>
          )}

          {/* ── Step 1: Open Eye EAR ─────────────────────────────────── */}
          {step === 1 && (
            <div className="space-y-4 text-center">
              <div className="text-6xl">👁️</div>
              <p className="text-slate-800 dark:text-slate-100 font-semibold text-lg">Open Eye Baseline</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Look straight at the camera with eyes fully open. Click Capture — system records 3 seconds of EAR data.</p>
              <div className="bg-slate-50 dark:bg-slate-800/40 rounded-xl p-3 border border-slate-200 dark:border-slate-800">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-0.5">Live EAR</p>
                <p className="text-3xl font-mono font-bold text-teal-700 dark:text-teal-400">{currentEar.toFixed(4)}</p>
              </div>
              {earOpenResult !== null && (
                <div className="bg-teal-50 dark:bg-teal-950/20 border border-teal-200 dark:border-teal-900/40 rounded-xl p-3">
                  <p className="text-xs text-teal-600 dark:text-teal-400">Captured open EAR</p>
                  <p className="text-2xl font-mono font-bold text-teal-800 dark:text-teal-300">{earOpenResult.toFixed(4)}</p>
                </div>
              )}
              <div className="flex gap-2">
                <button onClick={() => handleEarCapture('open')} disabled={capturing || !isCameraRunning}
                  className="flex-1 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white p-2.5 rounded-xl font-medium transition-colors">
                  {capturing ? '⏳ Capturing…' : '📸 Capture (3 s)'}
                </button>
                {earOpenResult !== null && (
                  <button onClick={() => { setCaptureError(''); setStep(2); }}
                    className="flex-1 bg-slate-700 hover:bg-slate-800 text-white p-2.5 rounded-xl font-medium transition-colors">
                    Next →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ── Step 2: Closed Eye EAR ───────────────────────────────── */}
          {step === 2 && (
            <div className="space-y-4 text-center">
              <div className="text-6xl">😑</div>
              <p className="text-slate-800 dark:text-slate-100 font-semibold text-lg">Closed Eye Baseline</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">Gently close both eyes. Click Capture — system records 3 seconds of EAR data.</p>
              <div className="bg-slate-50 dark:bg-slate-800/40 rounded-xl p-3 border border-slate-200 dark:border-slate-800">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-0.5">Live EAR</p>
                <p className="text-3xl font-mono font-bold text-teal-700 dark:text-teal-400">{currentEar.toFixed(4)}</p>
              </div>
              {earClosedResult !== null && (
                <div className="bg-teal-50 dark:bg-teal-950/20 border border-teal-200 dark:border-teal-900/40 rounded-xl p-3">
                  <p className="text-xs text-teal-600 dark:text-teal-400">Captured closed EAR</p>
                  <p className="text-2xl font-mono font-bold text-teal-800 dark:text-teal-300">{earClosedResult.toFixed(4)}</p>
                </div>
              )}
              <div className="flex gap-2">
                <button onClick={() => setStep(1)}
                  className="px-4 py-2.5 border border-slate-300 dark:border-slate-750 rounded-xl text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 text-sm font-medium">
                  ← Back
                </button>
                <button onClick={() => handleEarCapture('closed')} disabled={capturing || !isCameraRunning}
                  className="flex-1 bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white p-2.5 rounded-xl font-medium transition-colors">
                  {capturing ? '⏳ Capturing…' : '📸 Capture (3 s)'}
                </button>
                {earClosedResult !== null && (
                  <button onClick={() => { setCaptureError(''); handleStartBlinkCal(); }}
                    className="flex-1 bg-slate-700 hover:bg-slate-800 text-white p-2.5 rounded-xl font-medium transition-colors">
                    Next →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ── Step 3: Dot Blinks ───────────────────────────────────── */}
          {step === 3 && (
            <div className="space-y-4 text-center">
              <div className="text-5xl font-mono text-teal-600 dark:text-teal-400 tracking-wider">· · ·</div>
              <p className="text-slate-800 dark:text-slate-100 font-semibold text-lg">Short Blinks (Dots)</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Do <strong>{blinkCount}</strong> short, quick blinks — each under ~300 ms.
              </p>
              <div className="bg-slate-50 dark:bg-slate-800/40 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Progress</p>
                <p className="text-4xl font-mono font-bold text-teal-700 dark:text-teal-400">{dotCurrent} / {blinkCount}</p>
                <div className="mt-3 h-2.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-teal-500 rounded-full transition-all duration-300"
                    style={{ width: `${(dotCurrent / blinkCount) * 100}%` }} />
                </div>
              </div>
              <button onClick={handleNextStep}
                className="w-full border border-slate-300 dark:border-slate-750 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 p-2 rounded-xl text-sm transition-colors">
                Skip to Dash phase →
              </button>
            </div>
          )}

          {/* ── Step 4: Dash Blinks ──────────────────────────────────── */}
          {step === 4 && (
            <div className="space-y-4 text-center">
              <div className="text-4xl font-black text-slate-700 dark:text-slate-300 tracking-[0.3em]">— — —</div>
              <p className="text-slate-800 dark:text-slate-100 font-semibold text-lg">Long Blinks (Dashes)</p>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Do <strong>{blinkCount}</strong> long blinks — hold each blink for ~500 ms or more.
              </p>
              <div className="bg-slate-50 dark:bg-slate-800/40 rounded-xl p-4 border border-slate-200 dark:border-slate-800">
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Progress</p>
                <p className="text-4xl font-mono font-bold text-amber-600 dark:text-amber-400">{dashCurrent} / {blinkCount}</p>
                <div className="mt-3 h-2.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 rounded-full transition-all duration-300"
                    style={{ width: `${(dashCurrent / blinkCount) * 100}%` }} />
                </div>
              </div>
              <button onClick={handleNextStep}
                className="w-full border border-slate-300 dark:border-slate-750 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 p-2 rounded-xl text-sm transition-colors">
                Finalize now →
              </button>
            </div>
          )}

          {/* ── Step 5: Complete ─────────────────────────────────────── */}
          {step === 5 && (
            <div className="space-y-4">
              <div className="text-center">
                <div className="text-5xl mb-2">✅</div>
                <p className="text-slate-800 dark:text-slate-100 font-bold text-lg">Calibration Complete!</p>
              </div>
              <div className="bg-teal-50 dark:bg-teal-950/20 border border-teal-100 dark:border-teal-900/40 rounded-xl p-4 space-y-2 text-sm">
                <div className="flex justify-between text-teal-700 dark:text-teal-400">
                  <span>· Dot average</span><span className="font-mono font-bold">{calData.dotMs} ms</span>
                </div>
                <div className="flex justify-between text-teal-700 dark:text-teal-400">
                  <span>— Dash average</span><span className="font-mono font-bold">{calData.dashMs} ms</span>
                </div>
                <div className="flex justify-between text-teal-800 dark:text-teal-300 font-semibold border-t border-teal-200 dark:border-teal-900/40 pt-2">
                  <span>Threshold</span><span className="font-mono">{calData.thresholdMs} ms</span>
                </div>
                {earOpenResult !== null && earClosedResult !== null && (
                  <>
                    <div className="flex justify-between text-teal-700 dark:text-teal-400 border-t border-teal-200 dark:border-teal-900/40 pt-2">
                      <span>EAR open</span><span className="font-mono">{earOpenResult.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between text-teal-700 dark:text-teal-400">
                      <span>EAR closed</span><span className="font-mono">{earClosedResult.toFixed(4)}</span>
                    </div>
                  </>
                )}
              </div>
              <div>
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">Save as named profile (optional)</p>
                <div className="flex gap-2">
                  <input type="text" value={profileName} onChange={(e) => setProfileName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSaveProfile()}
                    placeholder="Profile name…"
                    className="flex-1 text-sm p-2 bg-white dark:bg-slate-850 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-800 dark:text-slate-100 focus:ring-1 focus:ring-teal-500 focus:outline-none" />
                  <button onClick={handleSaveProfile} disabled={savingProfile || !profileName.trim()}
                    className="bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white text-sm px-3 rounded-lg font-medium transition-colors">
                    {savingProfile ? '…' : 'Save'}
                  </button>
                </div>
              </div>
              <button onClick={onClose}
                className="w-full bg-slate-800 dark:bg-slate-700 hover:bg-slate-900 dark:hover:bg-slate-600 text-white p-2.5 rounded-xl font-semibold transition-colors">
                Done
              </button>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
function App() {
  // Theme state
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme');
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });

  useEffect(() => {
    console.log('BlinkLink Theme changed to:', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
      document.body.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
      document.body.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Auth
  const [currentUser, setCurrentUser] = useState(() => {
    const id = localStorage.getItem('userId');
    const name = localStorage.getItem('username');
    return id && name ? { id: parseInt(id, 10), username: name } : null;
  });

  // Settings sliders
  const [letterGap, setLetterGap] = useState(1.5);
  const [wordGap, setWordGap] = useState(3.0);
  const [sentenceGap, setSentenceGap] = useState(5.0);
  const [showMorseRef, setShowMorseRef] = useState(false);
  const [focusMode, setFocusMode] = useState(false);

  // Live AI metrics from WebSocket
  const [eyeState, setEyeState] = useState('UNKNOWN');
  const [confidence, setConfidence] = useState(0.0);
  const [fps, setFps] = useState(0.0);
  const [currentMorse, setCurrentMorse] = useState('');
  const [currentEar, setCurrentEar] = useState(0.0);
  const [decodedText, setDecodedText] = useState('');
  const [nlpSentences, setNlpSentences] = useState([]);  // [{raw, suggestions:[]}]
  const [selectedSuggestions, setSelectedSuggestions] = useState({});  // {sentenceIdx: choiceIdx}

  // Calibration state
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calProgress, setCalProgress] = useState(['DONE', 0, 0]);
  const [calData, setCalData] = useState({ isCalibrated: false, dotMs: 0, dashMs: 0, thresholdMs: 0 });
  const [calStatus, setCalStatus] = useState('Connecting to Engine…');
  const [showCalModal, setShowCalModal] = useState(false);

  // Camera
  const [isCameraRunning, setIsCameraRunning] = useState(false);

  // Calibration profiles (sidebar)
  const [calProfiles, setCalProfiles] = useState([]);
  const [showProfiles, setShowProfiles] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [profileBusy, setProfileBusy] = useState(false);

  // Pending config debounce
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
        console.error('Failed to update config', e);
      } finally {
        setPendingConfig({});
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [pendingConfig]);

  // WebSocket
  useEffect(() => {
    if (!currentUser) return;
    const wsUrl = API_BASE.replace(/^http/, 'ws') + '/ws/data';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setCalStatus(prev => prev.startsWith('✅') ? prev : 'System Online — Connected to Engine');
    };

    ws.onmessage = (event) => {
      try {
        const d = JSON.parse(event.data);
        if (d.eyeState) setEyeState(d.eyeState);
        if (d.confidence !== undefined) setConfidence(d.confidence);
        if (d.fps !== undefined) setFps(d.fps);
        if (d.morseSequence !== undefined) setCurrentMorse(d.morseSequence);
        if (d.decodedText !== undefined) setDecodedText(d.decodedText);
        if (d.ear !== undefined) setCurrentEar(d.ear);
        if (d.nlpSentences !== undefined) setNlpSentences(d.nlpSentences);

        if (d.isCalibrating !== undefined) setIsCalibrating(d.isCalibrating);
        if (d.calProgress !== undefined) setCalProgress(d.calProgress);

        if (d.isCalibrated !== undefined) {
          setCalData({
            isCalibrated: d.isCalibrated,
            dotMs: d.calDotMs ?? 0,
            dashMs: d.calDashMs ?? 0,
            thresholdMs: d.calThresholdMs ?? 0,
          });
        }

        if (d.isCalibrating) {
          const [phase, current, target] = d.calProgress;
          setCalStatus(`🎯 Calibrating ${phase}: ${current} / ${target} blinks`);
        } else if (d.isCalibrating === false) {
          setCalStatus(prev => prev.startsWith('🎯 Calibrating') ? '✅ Calibration Complete! Ready to type.' : prev);
        }
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onerror = () => setCalStatus('Connection Error! Is backend running?');
    ws.onclose = () => console.log('Disconnected from AI engine');

    return () => ws.close();
  }, [currentUser]);

  // ESC exits focus mode
  useEffect(() => {
    if (!focusMode) return;
    const onKey = (e) => { if (e.key === 'Escape') setFocusMode(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focusMode]);

  // ── Auth handlers ──────────────────────────────────────────────────────────

  const fetchProfiles = async () => {
    try {
      const res = await apiFetch('/api/calibrations');
      const data = await res.json();
      setCalProfiles(data.profiles || []);
    } catch (e) {
      console.error('Failed to fetch calibration profiles', e);
    }
  };

  const handleLogin = (user) => {
    localStorage.setItem('userId', String(user.id));
    localStorage.setItem('username', user.username);
    setCurrentUser({ id: user.id, username: user.username });
    setCalStatus(user.hasCalibration
      ? '✅ Calibration loaded from your account. Ready to type.'
      : 'Welcome! Open Calibration to set your blink thresholds.');
    setTimeout(fetchProfiles, 300);
  };

  const handleSignOut = () => {
    localStorage.removeItem('userId');
    localStorage.removeItem('username');
    setCurrentUser(null);
    setDecodedText(''); setNlpSentences([]); setCurrentMorse('');
    setEyeState('UNKNOWN'); setCalStatus('Connecting to Engine…');
  };

  // ── Slider ─────────────────────────────────────────────────────────────────

  const handleSliderChange = (key, setter) => (e) => {
    const val = parseFloat(e.target.value);
    setter(val);
    setPendingConfig(prev => ({ ...prev, [key]: val }));
  };

  // ── Camera ─────────────────────────────────────────────────────────────────

  const handleCameraStart = async () => {
    try {
      await apiFetch('/api/camera/start', { method: 'POST' });
      setIsCameraRunning(true);
      setCalStatus('📷 Camera starting…');
    } catch (e) { console.error('Failed to start camera'); }
  };

  const handleCameraStop = async () => {
    try {
      await apiFetch('/api/camera/stop', { method: 'POST' });
      setIsCameraRunning(false);
      setCalStatus('📷 Camera stopped.');
    } catch (e) { console.error('Failed to stop camera'); }
  };

  const handleCameraReset = async () => {
    try {
      await apiFetch('/api/camera/reset', { method: 'POST' });
      setDecodedText(''); setNlpSentences([]);
      setCalStatus('🔄 System & Camera Reset.');
    } catch (e) { console.error('Failed to reset camera'); }
  };

  // ── Text ───────────────────────────────────────────────────────────────────

  const handleClearText = async () => {
    setDecodedText(''); setNlpSentences([]); setSelectedSuggestions({});
    try { await apiFetch('/api/clear_text', { method: 'POST' }); }
    catch (e) { console.error('Failed to clear text'); }
  };

  // ── Calibration (sidebar load) ─────────────────────────────────────────────

  const handleLoadSavedCal = async () => {
    try {
      const res = await apiFetch('/api/calibration/load', { method: 'POST' });
      if (res.ok) setCalStatus('✅ Saved calibration reloaded.');
      else { const d = await res.json().catch(() => ({})); setCalStatus(`❌ ${d.detail || 'No saved calibration found.'}`); }
    } catch { setCalStatus('❌ Error loading saved calibration.'); }
  };

  const handleSaveProfile = async () => {
    if (!newProfileName.trim()) return;
    setProfileBusy(true);
    try {
      const res = await apiFetch('/api/calibrations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newProfileName.trim() }),
      });
      if (res.ok) { setNewProfileName(''); await fetchProfiles(); setCalStatus(`✅ Profile "${newProfileName.trim()}" saved.`); }
      else { const d = await res.json().catch(() => ({})); setCalStatus(`❌ ${d.detail || 'Failed to save profile.'}`); }
    } catch { setCalStatus('❌ Error saving profile.'); }
    finally { setProfileBusy(false); }
  };

  const handleLoadProfile = async (profileId, profileName) => {
    try {
      const res = await apiFetch(`/api/calibrations/${profileId}/load`, { method: 'POST' });
      if (res.ok) setCalStatus(`✅ Profile "${profileName}" loaded.`);
      else { const d = await res.json().catch(() => ({})); setCalStatus(`❌ ${d.detail || 'Failed to load profile.'}`); }
    } catch { setCalStatus('❌ Error loading profile.'); }
  };

  const handleDeleteProfile = async (profileId, profileName) => {
    try {
      const res = await apiFetch(`/api/calibrations/${profileId}`, { method: 'DELETE' });
      if (res.ok) { setCalProfiles(prev => prev.filter(p => p.id !== profileId)); setCalStatus(`🗑️ Profile "${profileName}" deleted.`); }
    } catch { setCalStatus('❌ Error deleting profile.'); }
  };

  const goFullscreen = () => {
    const el = document.documentElement;
    if (el.requestFullscreen) el.requestFullscreen().catch(() => {});
  };

  if (!currentUser) return <Login onLogin={handleLogin} />;

  return (
    <>
      <div className="flex h-screen bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 font-sans transition-colors duration-300">

        {/* ── SIDEBAR ──────────────────────────────────────────────── */}
        <div className="w-80 bg-white dark:bg-slate-900 p-6 overflow-y-auto border-r border-slate-200 dark:border-slate-800 flex flex-col shadow-sm z-10 text-slate-800 dark:text-slate-200">
          <h2 className="text-xl font-bold mb-6 text-slate-800 dark:text-slate-100">⚙️ Settings</h2>

          {/* Timing sliders */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">Letter Gap: {letterGap.toFixed(2)}s</label>
            <input type="range" min="0.3" max="5" step="0.1" value={letterGap}
              onChange={handleSliderChange('letter_gap_seconds', setLetterGap)}
              className="w-full accent-teal-600" />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">Word Gap: {wordGap.toFixed(2)}s</label>
            <input type="range" min="0.5" max="8" step="0.1" value={wordGap}
              onChange={handleSliderChange('word_gap_seconds', setWordGap)}
              className="w-full accent-teal-600" />
          </div>
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">Sentence Gap: {sentenceGap.toFixed(2)}s</label>
            <input type="range" min="1" max="12" step="0.1" value={sentenceGap}
              onChange={handleSliderChange('sentence_gap_seconds', setSentenceGap)}
              className="w-full accent-teal-600" />
          </div>

          <hr className="border-slate-100 dark:border-slate-800 my-4" />

          {/* Calibration */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-2 text-slate-800 dark:text-slate-100">🎯 Calibration</h3>

            {/* Active calibration values */}
            {calData.isCalibrated && (
              <div className="bg-teal-50 dark:bg-teal-950/20 border border-teal-100 dark:border-teal-900/40 rounded-lg p-3 mb-3 text-xs space-y-1">
                <p className="font-semibold text-teal-800 dark:text-teal-300 mb-1">Active Calibration</p>
                <div className="flex justify-between text-teal-700 dark:text-teal-400"><span>· Dot avg</span><span className="font-mono">{calData.dotMs} ms</span></div>
                <div className="flex justify-between text-teal-700 dark:text-teal-400"><span>— Dash avg</span><span className="font-mono">{calData.dashMs} ms</span></div>
                <div className="flex justify-between text-teal-800 dark:text-teal-300 font-semibold border-t border-teal-200 dark:border-teal-900/40 pt-1 mt-1"><span>Threshold</span><span className="font-mono">{calData.thresholdMs} ms</span></div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2 mb-2">
              <button onClick={() => setShowCalModal(true)}
                className="bg-teal-600 hover:bg-teal-700 text-white transition-colors p-2 rounded text-sm font-medium shadow-sm">
                Open Cal.
              </button>
              <button onClick={handleLoadSavedCal}
                className="bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 transition-colors p-2 rounded text-sm font-medium">
                Load Saved
              </button>
            </div>

            {/* Save as named profile */}
            {calData.isCalibrated && (
               <div className="mb-3">
                 <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Save current as profile</p>
                 <div className="flex gap-1">
                   <input type="text" value={newProfileName} onChange={(e) => setNewProfileName(e.target.value)}
                     onKeyDown={(e) => e.key === 'Enter' && handleSaveProfile()}
                     placeholder="Profile name…"
                     className="flex-1 text-xs p-1.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded text-slate-800 dark:text-slate-100 focus:ring-1 focus:ring-teal-500 focus:outline-none" />
                   <button onClick={handleSaveProfile} disabled={profileBusy || !newProfileName.trim()}
                     className="bg-teal-600 hover:bg-teal-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white text-xs px-2 rounded font-medium transition-colors">
                     {profileBusy ? '…' : 'Save'}
                   </button>
                 </div>
               </div>
            )}

            {/* Saved profiles */}
            <button onClick={() => { setShowProfiles(v => !v); if (!showProfiles) fetchProfiles(); }}
              className="w-full flex justify-between items-center text-xs text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-3 py-2 font-medium transition-colors">
              <span>Saved Profiles ({calProfiles.length})</span>
              <span className={`transform transition-transform ${showProfiles ? 'rotate-180' : ''}`}>▾</span>
            </button>

            {showProfiles && (
              <div className="mt-1 space-y-1 max-h-48 overflow-y-auto">
                {calProfiles.length === 0
                  ? <p className="text-xs text-slate-400 dark:text-slate-500 text-center py-2">No saved profiles yet</p>
                  : calProfiles.map(p => (
                    <div key={p.id} className="flex items-center gap-1 bg-slate-50 dark:bg-slate-800/30 border border-slate-100 dark:border-slate-800/50 rounded px-2 py-1.5">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-slate-700 dark:text-slate-300 truncate">{p.name}</p>
                        <p className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">·{Math.round(p.dot_ms)}ms  —{Math.round(p.dash_ms)}ms</p>
                      </div>
                      <button onClick={() => handleLoadProfile(p.id, p.name)}
                        className="text-[10px] bg-teal-600 hover:bg-teal-700 text-white px-1.5 py-0.5 rounded font-medium flex-shrink-0">Load</button>
                      <button onClick={() => handleDeleteProfile(p.id, p.name)}
                        className="text-[10px] bg-rose-100 dark:bg-rose-950/40 hover:bg-rose-200 dark:hover:bg-rose-900/50 text-rose-700 dark:text-rose-400 px-1.5 py-0.5 rounded font-medium flex-shrink-0">×</button>
                    </div>
                  ))}
              </div>
            )}
          </div>

          <hr className="border-slate-100 dark:border-slate-800 my-4" />

          {/* Text controls */}
          <div className="mt-auto">
            <h3 className="text-lg font-semibold mb-2 text-slate-800 dark:text-slate-100">📝 Text Controls</h3>
            <button onClick={handleClearText}
              className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-300 transition-colors p-2 rounded text-sm font-medium shadow-sm">
              Clear Text
            </button>
          </div>
        </div>

        {/* ── MAIN CONTENT ─────────────────────────────────────────── */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Header */}
          <header className="flex items-center justify-between p-6 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 z-10 transition-colors duration-300">
            <div className="flex items-center gap-4">
              <img src={blinklinkLogo} alt="BlinkLink Logo" className="w-12 h-12 rounded-xl shadow-md shadow-teal-500/20 object-cover" />
              <div>
                <h1 className="text-3xl font-extrabold text-slate-800 dark:text-slate-100">BlinkLink</h1>
                <p className="text-xs text-teal-600 dark:text-teal-400 tracking-widest uppercase font-bold">Assistive Communication</p>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm">
              <button
                onClick={() => {
                  console.log('Theme toggle button clicked! Current theme is:', theme);
                  setTheme(theme === 'dark' ? 'light' : 'dark');
                }}
                className="p-2 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-full font-medium transition-colors flex items-center justify-center w-8 h-8"
                aria-label="Toggle theme"
              >
                {theme === 'dark' ? '☀️' : '🌙'}
              </button>
              <span className="px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 rounded-full font-medium">
                Signed in as <span className="font-mono">{currentUser.username}</span>
              </span>
              <button onClick={handleSignOut}
                className="px-3 py-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-300 rounded-full font-medium transition-colors">
                Sign out
              </button>
            </div>
          </header>

          {/* Dashboard */}
          <div className="p-8 overflow-y-auto">
            <div className="grid grid-cols-12 gap-6 mb-8">

              {/* Live Video */}
              <div className="col-span-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 rounded-2xl shadow-sm">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-slate-800 dark:text-slate-100">
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
                  Live Video
                  <button onClick={() => setFocusMode(true)}
                    className="ml-auto text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700 px-3 py-1 rounded-full font-medium text-slate-700 dark:text-slate-300 transition-colors"
                    title="Expand video to fill the window">⛶ Focus Mode</button>
                </h3>
                <div className="w-full h-64 bg-slate-900 rounded-xl flex items-center justify-center mb-5 border border-slate-200 dark:border-slate-800 overflow-hidden relative shadow-inner">
                  <div className="absolute inset-0 border-2 border-teal-500/30 rounded-xl m-4 pointer-events-none z-10"></div>
                  <img src={`${API_BASE}/video_feed`} alt="Webcam Stream" className="w-full h-full object-cover" />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <button onClick={handleCameraStart} className="bg-teal-600 hover:bg-teal-700 text-white transition-colors p-2 rounded-lg font-medium shadow-sm">▶ Start</button>
                  <button onClick={handleCameraStop} className="bg-rose-500 hover:bg-rose-600 text-white transition-colors p-2 rounded-lg font-medium shadow-sm">⏹ Stop</button>
                  <button onClick={handleCameraReset} className="bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 transition-colors p-2 rounded-lg font-medium">🔄 Reset</button>
                </div>
              </div>

              {/* Status */}
              <div className="col-span-3 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 rounded-2xl shadow-sm flex flex-col gap-3">
                <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">📊 Status</h3>
                <div className="bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-100 dark:border-slate-800/60">
                  <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1 font-semibold">Eye State</p>
                  <p className="text-2xl font-mono text-teal-700 dark:text-teal-450 font-bold">{eyeState === 'OPEN' ? '👁️ ' : '😑 '}{eyeState}</p>
                </div>
                <div className="flex gap-4">
                  <div className="flex-1 bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-100 dark:border-slate-800/60">
                    <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1 font-semibold">Conf</p>
                    <p className="text-xl font-mono font-bold text-slate-700 dark:text-slate-300">{(confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div className="flex-1 bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl border border-slate-100 dark:border-slate-800/60">
                    <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1 font-semibold">FPS</p>
                    <p className="text-xl font-mono font-bold text-slate-700 dark:text-slate-300">{fps}</p>
                  </div>
                </div>
                <div className="bg-amber-50 dark:bg-amber-950/20 p-4 rounded-xl border border-amber-100 dark:border-amber-900/35">
                  <p className="text-xs text-amber-700 dark:text-amber-400 uppercase tracking-wider mb-1 font-bold">Current Morse</p>
                  <p className="text-3xl font-bold text-amber-600 dark:text-amber-400 tracking-[0.3em]">{currentMorse || <span className="text-amber-300/50 dark:text-amber-700/50 italic text-lg font-normal">none</span>}</p>
                </div>
              </div>

              {/* Text Output */}
              <div className="col-span-4 space-y-4 flex flex-col">

                {/* Raw decoded — current sentence only, cleared when sentence finishes */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 rounded-2xl shadow-sm flex flex-col" style={{ minHeight: '10rem' }}>
                  <h3 className="text-sm font-bold text-slate-600 dark:text-slate-350 uppercase tracking-wider mb-3">
                    Raw Decoded
                  </h3>
                  {(() => {
                    const parts = decodedText.split('\n\n');
                    const active = decodedText.endsWith('\n\n') ? '' : (parts.at(-1) || '');
                    return (
                      <p className="text-xl font-mono text-slate-700 dark:text-slate-300 leading-relaxed bg-slate-50 dark:bg-slate-800/40 p-4 rounded-xl flex-1 border border-slate-100 dark:border-slate-800/60 shadow-inner">
                        {active || <span className="text-slate-400 dark:text-slate-500 italic">Waiting for input…</span>}
                      </p>
                    );
                  })()}
                </div>

                {/* NLP corrected — immutable sentence history with 3 suggestions */}
                <div className="bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-950/20 dark:to-cyan-950/20 border border-teal-100 dark:border-teal-900/50 p-5 rounded-2xl shadow-sm flex flex-col relative overflow-hidden" style={{ minHeight: '10rem' }}>
                  <div className="absolute top-0 right-0 w-32 h-32 bg-teal-200/30 rounded-full blur-3xl pointer-events-none"></div>
                  <h3 className="text-sm font-bold text-teal-800 dark:text-teal-300 uppercase tracking-wider mb-3 relative z-10">
                    NLP Corrected History
                  </h3>
                  <div className="flex-1 overflow-y-auto space-y-3 relative z-10">
                    {nlpSentences.length === 0
                      ? <p className="text-teal-600/50 dark:text-teal-500/50 italic text-sm">Completed sentences will appear here…</p>
                      : (() => {
                          const total = nlpSentences.length;
                          const visibleCount = Math.min(total, 3);
                          const hiddenCount = total - visibleCount;
                          // newest first: take last 3 in reverse order
                          const visible = nlpSentences.slice(total - visibleCount).reverse();
                          return (
                            <>
                              {visible.map((sent, displayIdx) => {
                                const origIdx = total - 1 - displayIdx;
                                return (
                                  <div key={origIdx} className="bg-white/70 dark:bg-slate-900/60 border border-teal-100 dark:border-teal-900/40 rounded-xl p-3 shadow-sm">
                                    <p className="text-[10px] text-teal-600 dark:text-teal-400 uppercase font-bold tracking-wider mb-2">
                                      Sentence {origIdx + 1}
                                      <span className="ml-2 text-slate-400 dark:text-slate-500 normal-case font-normal">"{sent.raw}"</span>
                                    </p>
                                    <div className="space-y-1.5">
                                      {sent.suggestions.map((s, j) => (
                                        <button key={j}
                                          onClick={() => setSelectedSuggestions(prev => ({ ...prev, [origIdx]: j }))}
                                          className={`w-full text-left text-sm px-3 py-2 rounded-lg transition-colors border ${
                                            (selectedSuggestions[origIdx] ?? 0) === j
                                              ? 'bg-teal-600 text-white border-teal-600 font-semibold'
                                              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:border-teal-300 dark:hover:border-teal-800 hover:bg-teal-50 dark:hover:bg-teal-950/30'
                                          }`}>
                                          <span className="text-[10px] mr-1.5 opacity-60">{j === 0 ? '★' : `${j + 1}.`}</span>
                                          {s}
                                        </button>
                                      ))}
                                    </div>
                                  </div>
                                );
                              })}
                              {hiddenCount > 0 && (
                                <p className="text-center text-xs text-teal-600/50 py-1">
                                  … {hiddenCount} older sentence{hiddenCount > 1 ? 's' : ''} not shown
                                </p>
                              )}
                            </>
                          );
                        })()
                    }
                  </div>
                </div>

              </div>
            </div>

            {/* Calibration status bar */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 rounded-xl shadow-sm flex items-center gap-3">
              <span className="flex h-3 w-3 relative flex-shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
              </span>
              <p className="text-slate-600 dark:text-slate-300 font-mono text-sm tracking-wide font-medium">{calStatus}</p>
            </div>

            {/* Morse reference */}
            <div className="mt-4">
              <button onClick={() => setShowMorseRef(v => !v)} aria-expanded={showMorseRef}
                className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-850 transition-colors p-3 rounded-xl shadow-sm flex items-center justify-between text-slate-700 dark:text-slate-300 font-medium">
                <span className="flex items-center gap-2"><span className="text-amber-600 dark:text-amber-400">·−</span>Morse Code Reference</span>
                <span className={`transform transition-transform ${showMorseRef ? 'rotate-180' : ''}`}>▾</span>
              </button>
              {showMorseRef && (
                <div className="mt-3 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-sm p-5 space-y-5">
                  {Object.entries(MORSE_REFERENCE).map(([group, items]) => (
                    <div key={group}>
                      <h4 className="text-xs uppercase tracking-wider font-bold text-slate-500 dark:text-slate-400 mb-2">{group}</h4>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                        {items.map(([char, code]) => (
                          <div key={char} className="flex items-center justify-between bg-slate-50 dark:bg-slate-800/40 border border-slate-100 dark:border-slate-800/60 rounded-lg px-3 py-2">
                            <span className="font-mono font-bold text-slate-800 dark:text-slate-200">{char}</span>
                            <span className="font-mono text-amber-600 dark:text-amber-400 tracking-widest">{code}</span>
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

      {/* Focus Mode Overlay */}
      {focusMode && (
        <div className="fixed inset-0 z-50 bg-black">
          <img src={`${API_BASE}/video_feed`} alt="Webcam Stream (Focus)" className="absolute inset-0 w-full h-full object-cover" />
          <div className="absolute top-4 right-4 z-10 flex gap-2">
            <button onClick={goFullscreen} className="bg-white/10 hover:bg-white/20 backdrop-blur text-white text-sm px-3 py-2 rounded-lg border border-white/20 transition-colors">⛶ Go Fullscreen</button>
            <button onClick={() => setFocusMode(false)} className="bg-white/10 hover:bg-white/20 backdrop-blur text-white text-sm px-3 py-2 rounded-lg border border-white/20 transition-colors">✕ Exit Focus</button>
          </div>
          <div className="absolute bottom-0 inset-x-0 bg-black/70 backdrop-blur-sm text-white py-5 px-6 flex flex-col items-center gap-2 z-10">
            <p className="text-amber-400 tracking-[0.4em] text-3xl font-mono font-bold min-h-[1em]">{currentMorse || '·'}</p>
            <p className="text-xl font-mono text-center max-w-3xl break-words">
              {decodedText ? decodedText.slice(-40) : <span className="text-white/40 italic">Waiting for input…</span>}
            </p>
          </div>
        </div>
      )}

      {/* Calibration Modal */}
      {showCalModal && (
        <CalibrationModal
          onClose={() => setShowCalModal(false)}
          onStatusChange={setCalStatus}
          onCameraStart={handleCameraStart}
          isCameraRunning={isCameraRunning}
          isCalibrating={isCalibrating}
          calProgress={calProgress}
          calData={calData}
          currentEar={currentEar}
        />
      )}
    </>
  );
}

export default App;
