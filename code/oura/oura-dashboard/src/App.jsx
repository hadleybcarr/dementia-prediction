import { useEffect, useMemo, useState } from "react";

const API_URL =
  (import.meta?.env?.VITE_API_URL) ||
  (typeof process !== "undefined" && process.env?.REACT_APP_API_URL) ||
  "http://localhost:8000";

const DEFAULT_VITALS = {
  restingHR: "—",
  spo2:      "—",
  bodyTemp:  "—",
  respRate:  "—",
};

const VITAL_DEFS = [
  { key: "restingHR", label: "Resting Heart Rate",     unit: "bpm"  },
  { key: "temperature", label: "Temperature", unit: "degrees"},
  { key: "spo2",      label: "Blood Oxygen",           unit: "%"    },
  { key: "respRate",  label: "Respiratory Rate",       unit: "/min" },
];

const MODEL_DISPLAY = {
  cnn: "CNN",
  transformer: "Transformer",
  bilstm: "Bi-LSTM",
  ensemble: "Combined",
};


export default function VitalsDashboard() {
  const [data,       setData]       = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState(null);
  const [arch,       setArch]       = useState(null);  

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const [vRes, pRes] = await Promise.all([
          fetch(`${API_URL}/api/vitals`,  { cache: "no-store" }),
          fetch(`${API_URL}/api/predict`, { cache: "no-store" }),
        ]);
        if (!vRes.ok) throw new Error(`vitals HTTP ${vRes.status}`);
        const vJson = await vRes.json();
        const pJson = pRes.ok ? await pRes.json() : null;   
        if (!cancelled) {
          setData(vJson);
          setPrediction(pJson);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    const id = setInterval(load, 5 * 60 * 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);


  const perModel = prediction?.per_model ?? [];          
  const ensemble = prediction?.ensemble_risk ?? null;

  const archOptions = useMemo(() => {
    const opts = perModel.map(m => m.model);
    return ensemble != null ? ["ensemble", ...opts] : opts;
  }, [perModel.map(m => m.model).join(","), ensemble != null]);


  useEffect(() => {
    if (!arch && archOptions.length) setArch(archOptions[0]);
  }, [archOptions, arch]);


  const score = arch === "ensemble"
    ? (ensemble ?? 0)
    : (perModel.find(m => m.model === arch)?.risk ?? 0);
  const conf = Math.abs(2 * score - 1);

  const series = data?.hr_series ?? [];
  const hrVals = series.filter(v => v != null);
  const restingHr = hrVals.length ? Math.round(Math.min(...hrVals)) : null;

  const vitals = {
    restingHR:   restingHr            ?? DEFAULT_VITALS.restingHR,
    spo2:        data?.spo2           ?? DEFAULT_VITALS.spo2,
    temperature: data?.temperature    ?? DEFAULT_VITALS.bodyTemp,
    respRate:    data?.resp_rate      ?? DEFAULT_VITALS.respRate,
  };


  return (
    <div className="vd-root">
      <style>{styleSheet}</style>

      <div className="vd-shell">
        <h1 className="vd-title">
          Dementia&nbsp;Risk
        </h1>

        <p className="vd-subtitle vd-subtitle--center">
          The models below were trained on data from the MIMIC dataset. The
          dataset tends to skew towards older patients. Risk is relative to the patients in the dataset and you should consult a clinician 
          for an additional opinion. 
        </p>

        {/* status line — only shown if loading or errored */}
        {loading && !data && (
          <p className="vd-status">Loading live vitals…</p>
        )}
        {error && (
          <p className="vd-status vd-status--error">
            Couldn't reach backend: {error}. Showing placeholders.
          </p>
        )}

        <RiskHero score={score}/>

        <div className="vd-meta-pill">
          <Meta icon="model" label={MODEL_DISPLAY[arch] ?? arch ?? "—"} />
          <Meta icon="confidence" label={`${Math.round(conf * 100)}% confidence`} />
        </div>

        <div className="vd-arch-row">
          <span className="vd-arch-leader">— Model —</span>
          {archOptions.length === 0 && (
            <span className="vd-arch-leader">no models loaded</span>
          )}
          {archOptions.map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setArch(key)}
              className={`vd-arch ${arch === key ? "vd-arch--active" : ""}`}
            >
              {MODEL_DISPLAY[key] ?? key}
            </button>
          ))}
        </div>

        <SectionLabel>Vital Signs</SectionLabel>

        <div className="vd-vitals">
          {VITAL_DEFS.map((v) => (
            <article key={v.key} className="vd-vital">
              <p className="vd-vital-value">
                {vitals[v.key]}
                <span className="vd-vital-unit">{v.unit}</span>
              </p>
              <p className="vd-vital-label">{v.label}</p>
            </article>
          ))}
        </div>

        <p className="vd-footnote">
          This frontend is still under development.
        </p>
      </div>
    </div>
    
  );
}

function RiskHero({ score }) {
  const W = 880, H = 460;
  const CX = W / 2, CY = H / 2;
  const R = 168, RING_R = R + 18;
  const GOLD = "#B89048", CREAM = "#F4ECD0";
  const C = 2 * Math.PI * RING_R;
  const dash = C * Math.max(0, Math.min(1, score));

  return (
    <div className="vd-hero">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="vd-hero-svg"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Dementia risk score ${score.toFixed(2)}`}
      >
        <defs>
          <radialGradient id="vd-disc" cx="50%" cy="38%" r="68%">
            <stop offset="0%"   stopColor="#243F6E" />
            <stop offset="55%"  stopColor="#142C50" />
            <stop offset="100%" stopColor="#06152F" />
          </radialGradient>
          <linearGradient id="vd-gold-arc" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#8A6C2E" />
            <stop offset="50%"  stopColor="#E2C480" />
            <stop offset="100%" stopColor="#8A6C2E" />
          </linearGradient>
          <radialGradient id="vd-glass-hi" cx="50%" cy="20%" r="55%">
            <stop offset="0%"   stopColor="rgba(255,255,255,0.45)" />
            <stop offset="55%"  stopColor="rgba(255,255,255,0.06)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)"     />
          </radialGradient>
          <clipPath id="vd-disc-clip">
            <circle cx={CX} cy={CY} r={R} />
          </clipPath>
        </defs>

        <line x1="40" y1={CY} x2={CX - RING_R - 16} y2={CY}
              stroke={GOLD} strokeWidth="0.8" />
        <line x1={CX + RING_R + 16} y1={CY} x2={W - 40} y2={CY}
              stroke={GOLD} strokeWidth="0.8" />

        <circle cx={CX - RING_R - 30} cy={CY} r="5.5" fill={GOLD} />
        <circle cx={CX + RING_R + 30} cy={CY} r="5.5" fill={GOLD} />

        <circle cx="80" cy={CY} r="14" fill="none" stroke={GOLD} strokeWidth="1" />
        <circle cx={W - 80} cy={CY} r="14" fill="none" stroke={GOLD} strokeWidth="1" />

        <circle cx={CX} cy={CY} r={RING_R} fill="none"
                stroke={GOLD} strokeWidth="0.8" opacity="0.6" />

        <circle
          cx={CX} cy={CY} r={RING_R}
          fill="none" stroke="url(#vd-gold-arc)" strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${C - dash}`}
          transform={`rotate(-90 ${CX} ${CY})`}
        />

        <circle cx={CX} cy={CY} r={R} fill="url(#vd-disc)" />
        <ellipse cx={CX} cy={CY - R * 0.55} rx={R * 0.85} ry={R * 0.42}
                 fill="url(#vd-glass-hi)" clipPath="url(#vd-disc-clip)" />
        <circle cx={CX} cy={CY} r={R - 8} fill="none"
                stroke={GOLD} strokeWidth="0.5" opacity="0.45" />
        <circle cx={CX} cy={CY} r={R} fill="none"
                stroke="rgba(255,255,255,0.20)" strokeWidth="1" />

        <text
          x={CX} y={CY - 4}
          textAnchor="middle"
          fill={CREAM}
          style={{
            fontFamily: "'Cormorant Garamond', 'Playfair Display', Georgia, serif",
            fontSize: "108px",
            fontWeight: 500,
            letterSpacing: "-2px",
          }}
        >
          {score.toFixed(2)}
        </text>

        <text
          x={CX} y={CY + 44}
          textAnchor="middle"
          fill={CREAM}
          opacity="0.85"
          style={{
            fontFamily: "'Cormorant Garamond', Georgia, serif",
            fontStyle: "italic",
            fontSize: "22px",
            letterSpacing: "1.5px",
          }}
        >
          dementia risk
        </text>
      </svg>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <h2 className="vd-section">
      <span className="vd-section-bracket">{"{"}</span>
      <span className="vd-section-text">{children}</span>
      <span className="vd-section-bracket">{"}"}</span>
    </h2>
  );
}

function Meta({ icon, label }) {
  const paths = {
    model: (
      <>
        <path d="M7 1.5 L12.5 4.5 L12.5 9.5 L7 12.5 L1.5 9.5 L1.5 4.5 Z" />
        <path d="M7 1.5 L7 12.5 M1.5 4.5 L12.5 9.5 M12.5 4.5 L1.5 9.5" />
      </>
    ),
    confidence: (
      <path d="M7 1.7 L8.5 5 L12.2 5.5 L9.5 8 L10.2 11.6 L7 9.8 L3.8 11.6 L4.5 8 L1.8 5.5 L5.5 5 Z" />
    ),
  };
  return (
    <span className="vd-meta">
      <svg viewBox="0 0 14 14" width="13" height="13" className="vd-meta-icon" aria-hidden="true">
        {paths[icon]}
      </svg>
      <span>{label}</span>
    </span>
  );
}


const styleSheet = `
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500;1,600&family=Inter:wght@400;500;600&display=swap');

  .vd-root {
    --cream-bg:    #ffffff;
    --cream-card:  rgba(253, 250, 242, 0.55);
    --cream-solid: #ffffff;
    --ink:         #0E2240;
    --ink-soft:    #3D506E;
    --ink-mute:    #7C8FA8;
    --navy:        #0E2240;
    --navy-soft:   #1F3A66;
    --gold:        #B89048;
    --gold-bright: #D4B266;
    --gold-trace:  rgba(184, 144, 72, 0.30);
    --glass-hi:    rgba(255, 255, 255, 0.65);
    --glass-edge:  rgba(255, 255, 255, 0.85);

    background:
      radial-gradient(circle at 18% 12%, rgba(184,144,72,0.12), transparent 45%),
      radial-gradient(circle at 82% 88%, rgba(14,34,64,0.10),  transparent 50%),
      var(--cream-bg);
    color: var(--ink);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }


  .vd-title {
    font-family: 'Cormorant Garamond', 'Playfair Display', Georgia, serif;
    font-weight: 500;
    font-size: clamp(44px, 6vw, 76px);
    line-height: 1.04;
    letter-spacing: -1px;
    color: var(--ink);
    margin: 0 0 18px;
    text-align: center;
  }
  .vd-title em {
    font-style: italic;
    font-weight: 500;
    color: var(--gold);
    padding: 0 0.02em;
  }

  .vd-subtitle {
    max-width: 640px;
    margin: 0 0 56px;
    font-size: 15px;
    line-height: 1.7;
    color: var(--ink-soft);
  }
  .vd-subtitle--center {
    margin-left: auto;
    margin-right: auto;
    text-align: center;
  }

  .vd-status {
    text-align: center;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-mute);
    margin: -32px 0 32px;
  }
  .vd-status--error { color: #7A2D2D; }

  .vd-hero {
    margin: 0 0 24px;
    width: 100%;
    display: flex;
    justify-content: center;
  }
  .vd-hero-svg {
    width: 100%;
    height: auto;
  }

  .vd-meta-pill {
    margin: 0 auto 32px;
    width: max-content;
    display: flex;
    gap: 18px;
    align-items: center;
    padding: 10px 22px;
    border: 1px solid var(--gold-trace);
    border-radius: 999px;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.62), rgba(255,255,255,0.32));
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    box-shadow:
      inset 0 1px 0 var(--glass-edge),
      inset 0 -1px 0 rgba(184, 144, 72, 0.18),
      0 8px 24px rgba(14, 34, 64, 0.06);
  }
  .vd-meta {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    letter-spacing: 0.06em;
    color: var(--ink-soft);
    text-transform: uppercase;
  }
  .vd-meta-icon {
    fill: none;
    stroke: var(--gold);
    stroke-width: 1.3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .vd-arch-row {
    margin: 0 0 72px;
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
  }
  .vd-arch-leader {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 15px;
    color: var(--ink-mute);
    margin-right: 6px;
    letter-spacing: 0.03em;
  }
  .vd-arch {
    appearance: none;
    border: 1px solid var(--gold-trace);
    background: transparent;
    color: var(--ink-soft);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 9px 18px;
    border-radius: 999px;
    cursor: pointer;
    transition: all 180ms ease;
  }
  .vd-arch:hover {
    border-color: var(--navy-soft);
    color: var(--navy);
  }
  .vd-arch--active {
    background: linear-gradient(180deg, #1F3A66 0%, #0E2240 100%);
    border-color: var(--navy);
    color: var(--cream-solid);
    box-shadow:
      inset 0 1px 0 rgba(255, 255, 255, 0.18),
      0 4px 14px rgba(14, 34, 64, 0.18);
  }

  .vd-section {
    text-align: center;
    margin: 0 0 36px;
    font-family: 'Cormorant Garamond', serif;
    font-weight: 500;
    font-size: 30px;
    letter-spacing: 0.02em;
    color: var(--ink);
  }
  .vd-section-bracket {
    color: var(--gold);
    font-style: italic;
    font-size: 38px;
    line-height: 1;
    margin: 0 16px;
    font-weight: 400;
    vertical-align: -0.05em;
  }
  .vd-section-text { font-style: italic; }

  .vd-vitals {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 0 0 72px;
  }
  .vd-vital {
    position: relative;
    background:
      linear-gradient(155deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.38) 60%, rgba(212,178,102,0.10) 100%);
    border: 1px solid var(--gold-trace);
    border-radius: 16px;
    padding: 22px 22px 24px;
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    box-shadow:
      inset 0 1px 0 var(--glass-edge),
      inset 0 -1px 0 rgba(184, 144, 72, 0.20),
      0 10px 28px rgba(14, 34, 64, 0.06);
    transition: transform 240ms ease, border-color 240ms ease, box-shadow 240ms ease;
    overflow: hidden;
  }
  .vd-vital::before {
    content: '';
    position: absolute;
    top: -40%;
    left: -10%;
    width: 60%;
    height: 70%;
    background: radial-gradient(ellipse at center, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0) 65%);
    pointer-events: none;
    transform: rotate(-12deg);
  }
  .vd-vital:hover {
    transform: translateY(-3px);
    border-color: var(--gold-bright);
    box-shadow:
      inset 0 1px 0 var(--glass-edge),
      inset 0 -1px 0 rgba(184, 144, 72, 0.28),
      0 16px 36px rgba(14, 34, 64, 0.10);
  }
  .vd-vital-value {
    margin: 0;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-weight: 500;
    font-size: 46px;
    line-height: 1;
    color: var(--ink);
    letter-spacing: -0.5px;
  }
  .vd-vital-unit {
    font-size: 16px;
    font-style: italic;
    color: var(--ink-mute);
    margin-left: 4px;
  }
  .vd-vital-label {
    margin: 12px 0 0;
    font-size: 13px;
    color: var(--ink-soft);
    line-height: 1.4;
  }

  .vd-root {
  min-height: 100vh;
  width: 100%;
  padding: 56px 4vw;
  box-sizing: border-box;
}

  .vd-shell {
  width: 100%;
  margin: 0 auto;
  }

  .vd-footnote {
    margin: 0;
    text-align: center;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-mute);
  }
`;