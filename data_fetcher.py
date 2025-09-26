# data_fetcher.py
import requests
import streamlit as st

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


@st.cache_data(show_spinner=False, ttl=300)
def fetch_scoreboard(date: str):
    """Return list of events for the given YYYYMMDD date."""
    url = f"{ESPN_BASE}/scoreboard?dates={date}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json().get("events", [])
    except Exception as e:
        print("Error fetching scoreboard:", e)
        return []


@st.cache_data(show_spinner=False, ttl=300)
def fetch_team_details(team_id: str):
    """Fetch team details from ESPN (team metadata)."""
    url = f"{ESPN_BASE}/teams/{team_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Error fetching team details:", e)
        return {}


@st.cache_data(show_spinner=False, ttl=300)
def fetch_roster(team_id: str):
    """Fetch team roster JSON (raw)."""
    url = f"{ESPN_BASE}/teams/{team_id}/roster"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Error fetching roster:", e)
        return {}
