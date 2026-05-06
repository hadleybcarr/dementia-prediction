/**
 * Dashboard.jsx
 * -------------
 * Wraps your existing <VitalsDashboard /> with data fetching from /api/vitals.
 * Drop this in alongside VitalsDashboard.jsx and render <Dashboard /> from App.
 *
 * Expects FastAPI at http://localhost:8000 (see server.py).
 * Override with VITE_API_URL or REACT_APP_API_URL in .env if needed.
 */
import { useEffect, useState } from "react";

const API_URL =
  "http://localhost:8000";

export default function Dashboard(){
  const [data, setData]       = useState(null);
  const [error, setError]     = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const res = await fetch(`${API_URL}/api/vitals`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();

    // Re-poll every 5 minutes so the page stays fresh if it's left open.
    const id = setInterval(load, 5 * 60 * 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (loading) return <CenterMessage>Loading vitals…</CenterMessage>;
  if (error)            return <CenterMessage tone="error">Couldn't load vitals: {error}</CenterMessage>;
  if (!data)            return <CenterMessage>No data.</CenterMessage>;

  // Defensive defaults — if Oura returned null for any field, pass a sensible
  // fallback so the dashboard still renders rather than showing "null bpm".
  const safeVitals = {
    restingHR: data.vitals.restingHR ?? "—",
    hrv:       data.vitals.hrv       ?? "—",
    spo2:      data.vitals.spo2      ?? "—",
    bodyTemp:  data.vitals.bodyTemp  ?? "—",
    respRate:  data.vitals.respRate  ?? "—",
  };

  return (
    <Dashboard
      vitals={safeVitals}
      riskScores={data.riskScores}
      confidence={data.confidence}
    />
  );
}

function CenterMessage({ children, tone }) {
  return (
    <div
      style={{
        minHeight:  "100vh",
        display:    "grid",
        placeItems: "center",
        fontFamily: "'Cormorant Garamond', Georgia, serif",
        fontStyle:  "italic",
        fontSize:   22,
        color:      tone === "error" ? "#7a2d2d" : "#3D506E",
        background: "#EFEAE0",
        padding:    24,
        textAlign:  "center",
      }}
    >
      {children}
    </div>
  );
}