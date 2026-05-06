"""
server.py
=========
Unified Oura OAuth + Vitals API backend.

Usage:
  # 1. One-time interactive auth (opens browser, you paste the code back):
  python server.py auth

  # 2. Run the API:
  uvicorn server:app --reload --port 8000

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

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

CLIENT_ID     = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
REDIRECT_URI  = os.environ.get("REDIRECT_URI")

OURA_BASE     = "https://api.ouraring.com/v2/usercollection"
AUTHORIZE_URL = "https://cloud.ouraring.com/oauth/authorize"
TOKEN_URL     = "https://api.ouraring.com/oauth/token"

# Persisted token store. ADD tokens.json TO .gitignore!
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.json")

SCOPES        = "personal daily heartrate spo2 sleep heart_health"
LOOKBACK_DAYS = 7


# ── Token storage ─────────────────────────────────────────────────────────────
def save_tokens(access_token: str, refresh_token: str) -> None:
    with open(TOKEN_FILE, "w") as f:
        json.dump({
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "saved_at":      dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }, f, indent=2)
    try:
        os.chmod(TOKEN_FILE, 0o600)  # owner read/write only
    except OSError:
        pass  # non-POSIX filesystems


def load_tokens() -> Optional[Dict[str, str]]:
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE) as f:
        return json.load(f)


# ── OAuth flow (interactive, run once) ────────────────────────────────────────
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


# ── Oura helper with auto-refresh on 401 ──────────────────────────────────────
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


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Vitals Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _latest(items: list, time_keys=("bedtime_end", "day")) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    def stamp(x):
        for k in time_keys:
            if x.get(k):
                return x[k]
        return ""
    return max(items, key=stamp)


def fetch_vitals() -> Dict[str, Any]:
    today   = dt.date.today()
    start_d = (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_d   = today.isoformat()

    sleep     = oura_get("sleep",           {"start_date": start_d, "end_date": end_d})
    spo2      = oura_get("daily_spo2",      {"start_date": start_d, "end_date": end_d})
    readiness = oura_get("daily_readiness", {"start_date": start_d, "end_date": end_d})

    sleep_latest     = _latest(sleep.get("data", []))
    spo2_latest      = _latest(spo2.get("data", []),      time_keys=("day",))
    readiness_latest = _latest(readiness.get("data", []), time_keys=("day",))

    s = lambda f, d=None: (sleep_latest or {}).get(f, d)

    vitals = {
        "restingHR": s("lowest_heart_rate"),
        "hrv":       s("average_hrv"),
        "spo2":      ((spo2_latest or {}).get("spo2_percentage") or {}).get("average"),
        "bodyTemp":  (readiness_latest or {}).get("temperature_deviation"),
        "respRate":  s("average_breath"),
    }

    return {
        "vitals":     vitals,
        "riskScores": {"CNN": 0.78, "LSTM": 0.72, "Transformer": 0.81, "SVM": 0.65},
        "confidence": {"CNN": 0.92, "LSTM": 0.88, "Transformer": 0.94, "SVM": 0.79},
        "as_of":      dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


@app.get("/api/vitals")
def get_vitals():
    print("fetching vitals...")
    return fetch_vitals()


@app.get("/api/health")
def health():
    return {"ok": True, "tokens_present": load_tokens() is not None}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "auth":
        run_interactive_auth()
    else:
        print("Usage:")
        print("  python server.py auth                       # one-time OAuth")
        print("  uvicorn server:app --reload --port 8000     # run the API")