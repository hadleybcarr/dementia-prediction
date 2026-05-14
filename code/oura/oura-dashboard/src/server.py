"""
server.py
=========
Unified Oura OAuth + Vitals API backend.

Usage:
   1. python server.py auth
   2. uvicorn server:app --reload --port 8000
   3. curl http://localhost:8000/api/health
   4. curl http://localhost:8000/api/vitals
   5. curl http://localhost:8000/api/predict 

The access + refresh tokens are persisted to ./tokens.json. The vitals
endpoint auto-refreshes the access token on 401 using the refresh token,
so you only need to re-run `python server.py auth` if the refresh token
itself expires (rare).

Required in .env:
  CLIENT_ID
  CLIENT_SECRET
  REDIRECT_URI    e.g. http://localhost:5173/   (must match what's
                  registered in your Oura developer app)
"""

import os
import sys
import json
import datetime as dt
import webbrowser
from urllib.parse import urlencode
from typing import Any, Dict, Optional
import torch 

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

TRAIN_DIR = Path(__file__).resolve().parent / "../../../"
sys.path.insert(0, str(TRAIN_DIR.resolve()))
from train import load_model

load_dotenv()

CLIENT_ID     = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
REDIRECT_URI  = os.environ.get("REDIRECT_URI")
OURA_BASE     = "https://api.ouraring.com/v2/usercollection"
AUTHORIZE_URL = "https://cloud.ouraring.com/oauth/authorize"
TOKEN_URL     = "https://api.ouraring.com/oauth/token"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.json")
SCOPES        = "personal daily heartrate spo2 sleep"
LOOKBACK_DAYS = 7
CKPT_DIR = Path("../../../checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_AGE = 21
DEFAULT_SEX = 0
MODEL_NAMES = ["cnn", "transformer", "bilstm"]
FEATURE_ORDER = ["heart_rate", "spo2", "resp_rate", "temperature"]
MODELS = {}
T_HOURS = 24


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield

app = FastAPI(title="Vitals Dashboard API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

def save_tokens(access_token: str, refresh_token: str) -> None:
    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "saved_at":      dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }, f, indent=2)
    try:
        os.chmod(TOKEN_FILE, 0o600) 
    except OSError:
        pass 


def load_tokens() -> Optional[Dict[str, str]]:
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE) as f:
        return json.load(f)


def run_interactive_auth() -> None:
    if not (CLIENT_ID and CLIENT_SECRET and REDIRECT_URI):
        sys.exit("CLIENT_ID, CLIENT_SECRET, REDIRECT_URI must all be set in .env")

    auth_params = {
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         SCOPES,
    }
    auth_url = f"{AUTHORIZE_URL}?{urlencode(auth_params)}"
    print(f"\nOpen this URL to authorize:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    code = input("Paste the authorization code from the redirect URL: ").strip()

    resp = requests.post(TOKEN_URL, data={
        "grant_type":    "authorization_code",
        "code":          code,
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
    })
    if not resp.ok:
        sys.exit(f"Token exchange failed [{resp.status_code}]: {resp.text}")

    tokens = resp.json()
    save_tokens(tokens["access_token"], tokens["refresh_token"])
    print(f"\nTokens saved → {TOKEN_FILE}")


def refresh_access_token() -> str:
    """Refresh using the stored refresh_token. Updates tokens.json."""
    tokens = load_tokens()
    if not tokens:
        raise RuntimeError("No tokens.json — run `python server.py auth` first.")

    resp = requests.post(TOKEN_URL, data={
        "grant_type":    "refresh_token",
        "refresh_token": tokens["refresh_token"],
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })
    if not resp.ok:
        raise RuntimeError(f"Refresh failed [{resp.status_code}]: {resp.text}")

    new_tokens = resp.json()
    save_tokens(new_tokens["access_token"], new_tokens["refresh_token"])
    return new_tokens["access_token"]


def oura_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    tokens = load_tokens()
    if not tokens:
        raise HTTPException(401, "No tokens — run `python server.py auth` first")

    def _call(token: str) -> requests.Response:
        return requests.get(
            f"{OURA_BASE}/{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=15,
        )

    r = _call(tokens["access_token"])
    if r.status_code == 401:
        try:
            new_token = refresh_access_token()
        except RuntimeError as e:
            raise HTTPException(401, str(e))
        r = _call(new_token)

    if not r.ok:
        raise HTTPException(r.status_code, f"Oura {path}: {r.text}")
    return r.json()

def _latest(items: list, time_keys=("bedtime_end", "day")) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    def stamp(x):
        for k in time_keys:
            if x.get(k):
                return x[k]
        return ""
    return max(items, key=stamp)

def _nan_to_none(arr):
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
            for x in arr]

def fetch_hourly_window() -> dict:
    # 1. timezone-aware "now"; utcnow() is deprecated in 3.12+ (you're on 3.14)
    now_utc   = dt.datetime.now(dt.timezone.utc)
    start_utc = now_utc - dt.timedelta(hours=T_HOURS + 6)   # small buffer

    today   = dt.date.today()
    start_d = (today - dt.timedelta(days=2)).isoformat()
    end_d   = today.isoformat()

    hr = oura_get("heartrate", {
        "start_datetime": start_utc.isoformat(),
        "end_datetime":   now_utc.isoformat(),
    })
    hr_points = hr.get("data", [])
    hr_series = np.full(T_HOURS, np.nan, dtype=np.float32)

    if hr_points:
        end_hour = now_utc.replace(minute=0, second=0, microsecond=0)
        buckets  = [[] for _ in range(T_HOURS)]                 # list-of-lists is cleaner than dict
        for p in hr_points:
            ts = dt.datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00"))
            delta_h = int((end_hour - ts).total_seconds() // 3600)
            if 0 <= delta_h < T_HOURS:
                buckets[T_HOURS - 1 - delta_h].append(p["bpm"])
        for h, vals in enumerate(buckets):
            if vals:
                hr_series[h] = float(np.mean(vals))

    spo2_resp = oura_get("daily_spo2",      {"start_date": start_d, "end_date": end_d})
    sleep     = oura_get("sleep",           {"start_date": start_d, "end_date": end_d})
    readiness = oura_get("daily_readiness", {"start_date": start_d, "end_date": end_d})

    spo2_latest      = _latest(spo2_resp.get("data", []), time_keys=("day",)) or {}
    sleep_latest     = _latest(sleep.get("data", []))                          or {}
    readiness_latest = _latest(readiness.get("data", []), time_keys=("day",))  or {}  # 2. missing `or {}`

    return {
        "hr_series":   _nan_to_none(hr_series.tolist()),
        "spo2":        (spo2_latest.get("spo2_percentage") or {}).get("average"),
        "resp_rate":   sleep_latest.get("average_breath"),
        "temperature": readiness_latest.get("temperature_deviation"),
        "hr_coverage": int(np.sum(~np.isnan(hr_series))),       # 3. diagnostic — how many of 24h had data
    }


def _load_models():
    for name in MODEL_NAMES:
        ckpt_path = CKPT_DIR / f"best_{name}.pt"
        if not ckpt_path.exists():
            print(f"  · skip {name}: {ckpt_path} missing")
            continue
        model, meta = load_model(name, str(ckpt_path), device=DEVICE)
        MODELS[name] = {"model": model, "meta": meta}
    if not MODELS:
        print("WARNING: no checkpoints loaded — /api/predict will 503")

def _impute(series, fallback):
    arr = np.asarray(series, dtype=np.float32)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, fallback)
    # forward-fill then back-fill
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    arr = arr[idx]
    arr = np.where(np.isnan(arr), fallback, arr)
    return arr

def build_input(window: dict, meta: dict) -> torch.Tensor:
    fallbacks = {"heart_rate": 80.0, "spo2": 97.0, "resp_rate": 14.0, "temperature": 98.6}

    raw = {
        "heart_rate":  _impute(window["hr_series"], fallbacks["heart_rate"]),
        "spo2":        np.full(T_HOURS, window["spo2"]      or fallbacks["spo2"],        np.float32),
        "resp_rate":   np.full(T_HOURS, window["resp_rate"] or fallbacks["resp_rate"],   np.float32),
        "temperature": np.full(T_HOURS, window["temperature"] if window["temperature"] is not None else fallbacks["temperature"], np.float32),
    }
    bounds = meta.get("vital_bounds", {
        "heart_rate":  [20, 250],
        "spo2":        [50, 100],
        "resp_rate":   [4, 60],
        "temperature": [97.8, 99.1],
    })
    cols = []
    for name in meta.get("vital_names", ["heart_rate", "spo2", "resp_rate", "temperature"]):
        lo, hi = bounds[name]
        v = np.clip(raw[name], lo, hi)
        cols.append((v - lo) / (hi - lo))

    age_lo, age_hi = meta.get("age_bounds", [18, 100])
    age_scaled = (np.clip(DEFAULT_AGE, age_lo, age_hi) - age_lo) / (age_hi - age_lo)
    cols.append(np.full(T_HOURS, age_scaled, np.float32))
    cols.append(np.full(T_HOURS, float(DEFAULT_SEX),  np.float32))  # already in {0, 0.5, 1}

    x = np.stack(cols, axis=1)
    print("input is", x)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)


@app.get("/api/vitals")
def get_vitals():
    print("fetching vitals...")
    return fetch_hourly_window()

@app.get("/api/predict")
def predict():
    if not MODELS:
        raise HTTPException(503, "No model checkpoints loaded")
    window = fetch_hourly_window()
    per_model = []
    with torch.no_grad():
        for name, bundle in MODELS.items():
            x = build_input(window, bundle["meta"])
            logits = bundle["model"](x)
            p = float(torch.sigmoid(logits).squeeze().item())
            per_model.append({"model": name, "risk": p})
    ensemble = float(np.mean([m["risk"] for m in per_model]))
    return {
        "ensemble_risk": ensemble,
        "per_model": per_model,
        "caveats": {
            "missing_features": ["sbp", "dbp"],
            "domain": "model trained on MIMIC-IV hospital data; Oura inference is out-of-distribution",
        },
    }

@app.get("/api/health")
def health():
    return {"ok": True, "tokens_present": load_tokens() is not None}


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "auth":
        run_interactive_auth()
    else:
        print("Usage:")
        print("  python server.py auth                       # one-time OAuth")
        print("  uvicorn server:app --reload --port 8000     # run the API")