# data_fetcher.py
"""
Robust fetchers for ESPN public endpoints (scoreboard + roster).
These functions do NOT call Streamlit directly and only perform HTTP work when called.
They intentionally do minimal processing and return raw JSON (roster normalized in predictor/app).
"""

import requests
import datetime
from typing import Dict, Any

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

def get_scoreboard_for_date(date: datetime.date) -> Dict[str, Any]:
    """
    Fetch scoreboard JSON for the given date (YYYYMMDD). Raises exceptions on network errors.
    """
    datestr = date.strftime("%Y%m%d")
    url = f"{BASE}/scoreboard?dates={datestr}"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()

def get_team_roster(team_id: int) -> Dict[str, Any]:
    """
    Return the raw roster JSON from ESPN for the given team id.
    Caller is responsible for normalizing into a list of players.
    """
    url = f"{BASE}/teams/{team_id}/roster"
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return r.json()
