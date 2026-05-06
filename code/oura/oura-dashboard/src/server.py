"""
server.py
=========
Tiny FastAPI server that fetches Oura data and reshapes it for VitalsDashboard.

Run:
  pip install fastapi uvicorn requests python-dotenv
  uvicorn server:app --reload --port 5173

Endpoint:
  GET /api/vitals → {
      "vitals":      { restingHR, hrv, spo2, bodyTemp, respRate },
      "riskScores":  { CNN, LSTM, Transformer, SVM },
      "confidence":  { CNN, LSTM, Transformer, SVM },
      "as_of":       "2026-05-06T12:34:00Z"
  }

The access_token is read from .env at startup. After your first run of
auth.py, paste the token it printed into .env as OURA_ACCESS_TOKEN.

To run server: uvicorn server:app --reload --port 5173
"""

import os
import datetime as dt
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OURA_BASE        = "https://api.ouraring.com/v2/usercollection"
ACCESS_TOKEN     = os.environ.get("OURA_ACCESS_TOKEN")
LOOKBACK_DAYS    = 7   # how far back to ask Oura for; we still use the latest entry

app = FastAPI(title="Vitals Dashboard API")

# Allow your React dev server to call this. Tighten origins for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _oura_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not ACCESS_TOKEN:
        raise HTTPException(500, "OURA_ACCESS_TOKEN not set in .env")
    r = requests.get(
        f"{OURA_BASE}/{path}",
        headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
        params=params,
        timeout=15,
    )
    if not r.ok:
        raise HTTPException(r.status_code, f"Oura {path}: {r.text}")
    return r.json()


def _latest(items: list, key: str = "day") -> Optional[Dict[str, Any]]:
    """Return the most recent item from an Oura `data` list."""
    if not items:
        return None
    return max(items, key=lambda x: x.get(key) or x.get("bedtime_end") or "")


def fetch_vitals() -> Dict[str, Any]:
    today     = dt.date.today()
    start_d   = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_d     = today.isoformat()

    sleep_resp     = _oura_get("sleep",            {"start_date": start_d, "end_date": end_d})
    spo2_resp      = _oura_get("daily_spo2",       {"start_date": start_d, "end_date": end_d})
    readiness_resp = _oura_get("daily_readiness",  {"start_date": start_d, "end_date": end_d})

    sleep_latest     = _latest(sleep_resp.get("data", []),     key="bedtime_end")
    spo2_latest      = _latest(spo2_resp.get("data", []),      key="day")
    readiness_latest = _latest(readiness_resp.get("data", []), key="day")

    # Defensive lookups — log in dev if any of these come back as None.
    def s(field, default=None):
        return (sleep_latest or {}).get(field, default)

    def spo2_avg():
        if not spo2_latest:
            return None
        block = spo2_latest.get("spo2_percentage") or {}
        return block.get("average")

    def temp_dev():
        return (readiness_latest or {}).get("temperature_deviation")

    vitals = {
        "restingHR": s("lowest_heart_rate"),
        "hrv":       s("average_hrv"),
        "spo2":      spo2_avg(),
        "bodyTemp":  temp_dev(),
        "respRate":  s("average_breath"),
    }

    # Plug in your real model inference here. Hard-coded for now so the
    # dashboard renders end-to-end before models are wired up.
    risk_scores = {"CNN": 0.78, "LSTM": 0.72, "Transformer": 0.81, "SVM": 0.65}
    confidence  = {"CNN": 0.92, "LSTM": 0.88, "Transformer": 0.94, "SVM": 0.79}

    return {
        "vitals":     vitals,
        "riskScores": risk_scores,
        "confidence": confidence,
        "as_of":      dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


@app.get("/api/vitals")
def get_vitals():
    return fetch_vitals()


@app.get("/api/health")
def health():
    return {"ok": True, "token_set": bool(ACCESS_TOKEN)}