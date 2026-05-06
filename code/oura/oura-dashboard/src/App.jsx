import { useMemo, useState } from "react";

/**
 * VitalsDashboard
 * ----------------
 * Editorial dashboard combining Oura-streamed vitals with a multi-architecture
 * dementia-risk readout. Aesthetic: warm cream surface, deep navy & gold accents,
 * liquid-glass cards, and a "celestial axis" motif framing the central score.
 *
 * Default props render a polished standalone preview. Wire your own:
 *
 *   <VitalsDashboard
 *     vitals={{ restingHR, hrv, spo2, respRate }}
 *     riskScores={{ CNN, LSTM, Transformer, SVM }}
 *     confidence={{ CNN, LSTM, Transformer, SVM }}
 *   />
 *
 * Fonts (Cormorant Garamond + Inter) are loaded via @import inside the
 * component's <style> tag — no parent setup required.
 */

const DEFAULT_VITALS = {
  restingHR: 58,   // bpm
  hrv: 42,         // ms (rMSSD)
  spo2: 98,        // %
  bodyTemp: 0.2,   // °C deviation from baseline
  respRate: 14,    // breaths/min
};

const DEFAULT_RISKS = {
  CNN: 0.78,
  LSTM: 0.72,
  Transformer: 0.81,
  SVM: 0.65,
};

const DEFAULT_CONFIDENCE = {
  CNN: 0.92,
  LSTM: 0.88,
  Transformer: 0.94,
  SVM: 0.79,
};

const VITAL_DEFS = [
  { key: "restingHR", label: "Resting Heart Rate",      eyebrow: "cardiac",      unit: "bpm"  },
  { key: "hrv",       label: "Heart Rate Variability",  eyebrow: "autonomic",    unit: "ms"   },
  { key: "spo2",      label: "Blood Oxygen",            eyebrow: "respiratory",  unit: "%"    },
  { key: "respRate",  label: "Respiratory Rate",        eyebrow: "breath",       unit: "/min" },
];

const ARCHITECTURES = ["CNN", "LSTM", "Transformer", "SVM"];

export default function VitalsDashboard({
  vitals = DEFAULT_VITALS,
  riskScores = DEFAULT_RISKS,
  confidence = DEFAULT_CONFIDENCE,
}) {
  const [arch, setArch] = useState("CNN");
  const score = riskScores[arch] ?? 0;
  const conf = confidence[arch] ?? 0;

  const tier = useMemo(() => {
    if (score < 0.4) return "Low";
    if (score < 0.7) return "Moderate";
    return "Elevated";
  }, [score]);

  return (
    <div className="vd-root">
      <style>{styleSheet}</style>

      <div className="vd-shell">

        {/* display headline */}
        <h1 className="vd-title">
          <em>D</em>ementia&nbsp;<em>R</em>isk
        </h1>

       <center><p className="vd-subtitle">
          The models below were trained on data from the MIMIC dataset. The dataset tends to skew towards older patients.
      </p></center>

        {/* hero — risk circle with celestial line motif */}
        <RiskHero score={score} tier={tier} />

        {/* stats pill below the hero (mirrors the Venice / Max 20 / 3 hours pill) */}
        <div className="vd-meta-pill">
          <Meta icon="model"      label={arch} />
          <Meta icon="confidence" label={`${Math.round(conf * 100)}% confidence`} />
          <Meta icon="tier"       label={`${tier} tier`} />
        </div>

        {/* model selector */}
        <div className="vd-arch-row">
          <span className="vd-arch-leader">— Model —</span>
          {ARCHITECTURES.map((a) => (
            <button
              key={a}
              type="button"
              onClick={() => setArch(a)}
              className={`vd-arch ${arch === a ? "vd-arch--active" : ""}`}
            >
              {a}
            </button>
          ))}
        </div>

        {/* vitals */}
        <SectionLabel>Vital Signs</SectionLabel>

        <div className="vd-vitals">
          {VITAL_DEFS.map((v) => (
            <article key={v.key} className="vd-vital">
              <p className="vd-vital-eyebrow">{v.eyebrow}</p>
              <p className="vd-vital-value">
                {vitals[v.key]}
                <span className="vd-vital-unit">{v.unit}</span>
              </p>
              <p className="vd-vital-label">{v.label}</p>
            </article>
          ))}
        </div>

        <p className="vd-footnote">
          <em>Oura</em> data refreshed continuously &middot; model inference last
          run a few moments ago
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Hero — the central focal element                                    */
/* ------------------------------------------------------------------ */
function RiskHero({ score, tier }) {
  const W = 880;
  const H = 460;
  const CX = W / 2;
  const CY = H / 2;
  const R = 168;
  const RING_R = R + 18;

  const GOLD = "#B89048";
  const CREAM = "#F4ECD0";

  // arc proportional to score for the outer progress ring
  const C = 2 * Math.PI * RING_R;
  const dash = C * Math.max(0, Math.min(1, score));

  return (
    <div className="vd-hero">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="vd-hero-svg"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Dementia risk score ${score.toFixed(2)}, ${tier} tier`}
      >
        <defs>
          {/* deep navy disc with subtle radial depth */}
          <radialGradient id="vd-disc" cx="50%" cy="38%" r="68%">
            <stop offset="0%"   stopColor="#243F6E" />
            <stop offset="55%"  stopColor="#142C50" />
            <stop offset="100%" stopColor="#06152F" />
          </radialGradient>

          {/* metallic gold gradient for the progress arc */}
          <linearGradient id="vd-gold-arc" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#8A6C2E" />
            <stop offset="50%"  stopColor="#E2C480" />
            <stop offset="100%" stopColor="#8A6C2E" />
          </linearGradient>

          {/* liquid-glass top highlight: soft elliptical sheen near the top of the disc */}
          <radialGradient id="vd-glass-hi" cx="50%" cy="20%" r="55%">
            <stop offset="0%"   stopColor="rgba(255,255,255,0.45)" />
            <stop offset="55%"  stopColor="rgba(255,255,255,0.06)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0)"     />
          </radialGradient>

          {/* clip path so the highlight stays inside the disc */}
          <clipPath id="vd-disc-clip">
            <circle cx={CX} cy={CY} r={R} />
          </clipPath>
        </defs>

        {/* horizontal axis line: stops at the ring on each side */}
        <line x1="40" y1={CY} x2={CX - RING_R - 16} y2={CY}
              stroke={GOLD} strokeWidth="0.8" />
        <line x1={CX + RING_R + 16} y1={CY} x2={W - 40} y2={CY}
              stroke={GOLD} strokeWidth="0.8" />

        {/* near-circle filled markers */}
        <circle cx={CX - RING_R - 30} cy={CY} r="5.5" fill={GOLD} />
        <circle cx={CX + RING_R + 30} cy={CY} r="5.5" fill={GOLD} />

        {/* far-end outline circles */}
        <circle cx="80" cy={CY} r="14" fill="none" stroke={GOLD} strokeWidth="1" />
        <circle cx={W - 80} cy={CY} r="14" fill="none" stroke={GOLD} strokeWidth="1" />

        {/* faint outer architectural frame */}
        <circle cx={CX} cy={CY} r={RING_R} fill="none"
                stroke={GOLD} strokeWidth="0.8" opacity="0.6" />

        {/* progress arc proportional to the score (metallic gold) */}
        <circle
          cx={CX} cy={CY} r={RING_R}
          fill="none" stroke="url(#vd-gold-arc)" strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${C - dash}`}
          transform={`rotate(-90 ${CX} ${CY})`}
        />

        {/* the deep navy disc */}
        <circle cx={CX} cy={CY} r={R} fill="url(#vd-disc)" />

        {/* liquid-glass top sheen — clipped to the disc */}
        <ellipse cx={CX} cy={CY - R * 0.55} rx={R * 0.85} ry={R * 0.42}
                 fill="url(#vd-glass-hi)" clipPath="url(#vd-disc-clip)" />

        {/* inner gold ring tracing */}
        <circle cx={CX} cy={CY} r={R - 8} fill="none"
                stroke={GOLD} strokeWidth="0.5" opacity="0.45" />

        {/* outer rim highlight — the glass edge */}
        <circle cx={CX} cy={CY} r={R} fill="none"
                stroke="rgba(255,255,255,0.20)" strokeWidth="1" />

        {/* score readout — display serif in champagne cream */}
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

        {/* "dementia risk" italic caption inside the disc */}
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

/* ------------------------------------------------------------------ */
/* Bracketed section label — { Vital Signs }                           */
/* ------------------------------------------------------------------ */
function SectionLabel({ children }) {
  return (
    <h2 className="vd-section">
      <span className="vd-section-bracket">{"\u007B"}</span>
      <span className="vd-section-text">{children}</span>
      <span className="vd-section-bracket">{"\u007D"}</span>
    </h2>
  );
}

/* ------------------------------------------------------------------ */
/* Highlight ticket card                                               */
/* ------------------------------------------------------------------ */
function Highlight({ title, children }) {
  return (
    <article className="vd-highlight">
      <h3 className="vd-highlight-title">{title}</h3>
      <p className="vd-highlight-body">{children}</p>
    </article>
  );
}

/* ------------------------------------------------------------------ */
/* Meta pill segment with a tiny inline icon                           */
/* ------------------------------------------------------------------ */
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
    tier: <path d="M2 11 L5 7 L8 9 L12 3" />,
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

/* ================================================================== */
/* Styles                                                              */
/* ================================================================== */
const styleSheet = `
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500;1,600&family=Inter:wght@400;500;600&display=swap');

  /* ------- palette --------------------------------------------------*/
  .vd-root {
    --cream-bg:    #EFEAE0;
    --cream-card:  rgba(253, 250, 242, 0.55);
    --cream-solid: #F8F4E9;
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
    min-height: 100vh;
    width: 100%;
    padding: 56px 24px 80px;
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
  }

  .vd-shell {
    max-width: 1080px;
    margin: 0 auto;
  }

  /* ------- breadcrumb ----------------------------------------------*/
  .vd-crumb {
    margin: 0 0 28px;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-mute);
  }
  .vd-crumb em {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 14px;
    text-transform: none;
    letter-spacing: 0.04em;
    color: var(--gold);
  }
  .vd-sep { margin: 0 8px; color: var(--gold-trace); }

  /* ------- title ---------------------------------------------------*/
  .vd-title {
    font-family: 'Cormorant Garamond', 'Playfair Display', Georgia, serif;
    font-weight: 500;
    font-size: clamp(44px, 6vw, 76px);
    line-height: 1.04;
    letter-spacing: -1px;
    color: var(--ink);
    margin: 0 0 18px;
  }
  .vd-title em {
    font-style: italic;
    font-weight: 500;
    color: var(--gold);
    padding: 0 0.02em;
  }

  /* ------- subtitle ------------------------------------------------*/
  .vd-subtitle {
    max-width: 640px;
    margin: 0 0 56px;
    font-size: 15px;
    line-height: 1.7;
    color: var(--ink-soft);
  }
  .vd-subtitle em {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 17px;
    color: var(--gold);
  }

  /* ------- hero ----------------------------------------------------*/
  .vd-hero {
    margin: 0 0 24px;
    width: 100%;
    display: flex;
    justify-content: center;
  }
  .vd-hero-svg {
    width: 100%;
    height: auto;
    max-width: 880px;
  }

  /* ------- meta pill (liquid glass) --------------------------------*/
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

  /* ------- architecture selector -----------------------------------*/
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

  /* ------- bracketed section label ---------------------------------*/
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

  /* ------- vitals --------------------------------------------------*/
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
    /* specular highlight in the top-left corner — the "liquid glass" sheen */
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
  .vd-vital-eyebrow {
    margin: 0 0 10px;
    font-size: 10px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 600;
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

  /* ------- highlights ----------------------------------------------*/
  .vd-highlights {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin: 0 0 56px;
  }
  .vd-highlight {
    background:
      linear-gradient(155deg, rgba(255,255,255,0.70) 0%, rgba(255,255,255,0.36) 60%, rgba(31,58,102,0.08) 100%);
    border: 1px solid var(--gold-trace);
    border-radius: 16px;
    padding: 22px 24px 24px;
    position: relative;
    backdrop-filter: blur(14px) saturate(135%);
    -webkit-backdrop-filter: blur(14px) saturate(135%);
    box-shadow:
      inset 0 1px 0 var(--glass-edge),
      inset 0 -1px 0 rgba(184, 144, 72, 0.18),
      0 10px 28px rgba(14, 34, 64, 0.06);
  }
  .vd-highlight::before,
  .vd-highlight::after {
    content: '';
    position: absolute;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--cream-bg);
    top: 50%;
    transform: translateY(-50%);
    border: 1px solid var(--gold-trace);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
  }
  .vd-highlight::before { left: -8px; }
  .vd-highlight::after  { right: -8px; }
  .vd-highlight-title {
    margin: 0 0 10px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.02em;
    color: var(--ink);
  }
  .vd-highlight-body {
    margin: 0;
    font-size: 13px;
    line-height: 1.65;
    color: var(--ink-soft);
  }
  .vd-highlight-body em {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 15px;
    color: var(--gold);
    margin: 0 1px;
  }

  /* ------- footnote ------------------------------------------------*/
  .vd-footnote {
    margin: 0;
    text-align: center;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-mute);
  }
  .vd-footnote em {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 13px;
    text-transform: none;
    letter-spacing: 0.02em;
    color: var(--gold);
  }
`;