# app.py
"""
Streamlit UI for the improved NFL Predictor workshop app.
"""

import streamlit as st
import datetime
import pandas as pd
from typing import List, Dict

from data_fetcher import get_scoreboard_for_date, get_team_roster
from predictor import (compute_matchup_and_player_prediction,
                       ensure_and_train_demo_model,
                       get_default_baseline, DEMO_PLAYER_HISTORY)

st.set_page_config(page_title="NFL Predictor — Workshop (Improved)", layout="wide")
st.title("🏈 NFL Predictor — Workshop (Improved)")

# ---------- Cached fetch helpers ----------
@st.cache_data(ttl=120)
def fetch_scoreboard_cached(d: datetime.date) -> Dict:
    return get_scoreboard_for_date(d)

@st.cache_data(ttl=300)
def fetch_roster_cached(team_id: str) -> List[Dict]:
    try:
        j = get_team_roster(int(team_id))
        players = []
        if isinstance(j, dict):
            athletes = j.get("athletes")
            if isinstance(athletes, list) and athletes:
                for group in athletes:
                    if isinstance(group, dict) and isinstance(group.get("items"), list):
                        for item in group["items"]:
                            athlete = item.get("athlete") if isinstance(item, dict) and item.get("athlete") else item
                            if isinstance(athlete, dict):
                                name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("name")
                                pid = athlete.get("id") or athlete.get("guid") or athlete.get("uid")
                                pos = item.get("position") or athlete.get("position") or {}
                                pos_abbr = (pos.get("abbreviation") if isinstance(pos, dict) else pos) or ""
                                players.append({"id": str(pid or ""), "name": name or "Unknown", "position": (pos_abbr or "").upper()})
                    elif isinstance(group, dict):
                        athlete = group.get("athlete") or group
                        if isinstance(athlete, dict):
                            name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("name")
                            pid = athlete.get("id") or athlete.get("guid") or athlete.get("uid")
                            pos = (athlete.get("position") or {})
                            pos_abbr = (pos.get("abbreviation") if isinstance(pos, dict) else pos) or ""
                            players.append({"id": str(pid or ""), "name": name or "Unknown", "position": (pos_abbr or "").upper()})
        return players
    except Exception:
        return []

def tidy_record_summary(record_list):
    try:
        if not isinstance(record_list, list):
            return None
        for r in record_list:
            if (r.get("name") and r.get("name").lower() == "overall") or (r.get("type") and r.get("type").lower() == "total"):
                return r.get("summary") or r.get("displayValue")
        return ", ".join(str(r.get("summary") or r.get("displayValue") or "") for r in record_list if isinstance(r, dict))
    except Exception:
        return None

# ---------- Main ----------
st.markdown("**Model selection:** Choose which model to use for win probabilities & player predictions.")
model_choice = st.selectbox("Prediction model", ["heuristic", "elo", "regression"])

date = st.date_input("Pick a date", value=datetime.date.today())
with st.spinner("Loading scoreboard..."):
    try:
        scoreboard = fetch_scoreboard_cached(date)
        events = scoreboard.get("events", []) if scoreboard else []
    except Exception as e:
        st.error(f"Failed fetching scoreboard: {e}")
        st.stop()

if not events:
    st.info("No live games found for that date — using a demo matchup.")
    events = [{
        "id": "demo123",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"id": "1", "displayName": "Dallas Cowboys"}, "records": [{"summary":"2-1"}]},
                {"homeAway": "away", "team": {"id": "2", "displayName": "New York Giants"}, "records": [{"summary":"1-2"}]}
            ]
        }]
    }]

# Build selectable games list
game_labels = []
parsed_games = []
for ev in events:
    comp = ev.get("competitions", [{}])[0]
    comps = comp.get("competitors", [])
    if len(comps) < 2:
        continue
    home = next((c for c in comps if c.get("homeAway") == "home"), comps[0])
    away = next((c for c in comps if c.get("homeAway") == "away"), comps[1])
    label = f"{away.get('team',{}).get('displayName','Away')} @ {home.get('team',{}).get('displayName','Home')}"
    game_labels.append(label)
    parsed_games.append((home, away, ev))

if not game_labels:
    st.error("No games to display.")
    st.stop()

choice = st.selectbox("Choose a game", game_labels)
idx = game_labels.index(choice)
home, away, selected_event = parsed_games[idx]

col1, col2 = st.columns(2)
with col1:
    st.subheader(home.get("team", {}).get("displayName", "Home"))
    rec = tidy_record_summary(home.get("records", []))
    if rec:
        st.write("Record:", rec)
with col2:
    st.subheader(away.get("team", {}).get("displayName", "Away"))
    rec2 = tidy_record_summary(away.get("records", []))
    if rec2:
        st.write("Record:", rec2)

home_id = home.get("team", {}).get("id") or home.get("team", {}).get("uid") or "1"
away_id = away.get("team", {}).get("id") or away.get("team", {}).get("uid") or "2"

with st.spinner("Fetching rosters (cached)..."):
    home_players = fetch_roster_cached(str(home_id))
    away_players = fetch_roster_cached(str(away_id))

def roster_to_df(player_list):
    if not player_list:
        return pd.DataFrame([{"id": "0", "name": "Demo Player", "position": "QB"}])
    rows = []
    for p in player_list:
        rows.append({
            "id": p.get("id", ""),
            "name": p.get("name", "") or p.get("displayName", ""),
            "position": (p.get("position") or "").upper()
        })
    return pd.DataFrame(rows)

home_df = roster_to_df(home_players)
away_df = roster_to_df(away_players)

st.write("### Home roster (sample)")
st.dataframe(home_df.head(40))
st.write("### Away roster (sample)")
st.dataframe(away_df.head(40))

team_choice = st.radio("Pick which team's player you'd like to predict for",
                       [home.get("team", {}).get("displayName", "Home"),
                        away.get("team", {}).get("displayName", "Away")])

players_df = home_df if team_choice == home.get("team", {}).get("displayName", "Home") else away_df
player_options = players_df.apply(lambda r: f"{r['name']} — {r['position'] or 'UNK'}|{r['id']}", axis=1).tolist()
labels = [opt.split("|")[0] for opt in player_options] if player_options else ["Demo Player"]
player_choice_label = st.selectbox("Pick a player", labels)
selected_index = labels.index(player_choice_label)
selected_player_token = player_options[selected_index] if player_options else "Demo Player|0"
name_part, id_part = selected_player_token.split("|", 1) if "|" in selected_player_token else (selected_player_token, "0")
player_name = name_part.split(" — ")[0].strip()
player_pos = name_part.split(" — ")[1].strip() if " — " in name_part else "UNK"

stat_type = st.selectbox("Which stat to predict for the selected player?",
                         ["passing_yds", "rushing_yds", "receiving_yds", "fantasy_points"])

default_baseline = get_default_baseline(player_pos or "UNK", stat_type)
st.markdown(f"**Auto baseline (by position)** for {player_name}: **{default_baseline} {('yards/game' if 'yds' in stat_type else 'fantasy pts') }**")

# advanced override
with st.expander("Advanced: override baseline / usage (optional)"):
    override_baseline = st.number_input("Override baseline (yards)", value=float(default_baseline))
    usage = st.slider("Usage multiplier", 0.2, 2.0, 1.0, 0.05)

try:
    baseline_val = float(override_baseline)
    usage_val = float(usage)
except Exception:
    baseline_val = default_baseline
    usage_val = 1.0

home_stats = {"avg_points_scored": 23.0, "avg_points_allowed": 21.0, "defense_strength": 1.0}
away_stats = {"avg_points_scored": 20.0, "avg_points_allowed": 24.0, "defense_strength": 1.0}

if model_choice == "regression":
    with st.spinner("Preparing demo regression model..."):
        try:
            ensure_and_train_demo_model()
            st.success("Regression demo model prepared (if sklearn available).")
        except Exception as e:
            st.warning(f"Regression demo model prepare failed (non-fatal): {e}")

if st.button("Compute predictions"):
    player_profile = {
        "avg_stat": baseline_val,
        "stat_type": stat_type,
        "usage": usage_val,
        "position": player_pos
    }
    if team_choice == home.get("team", {}).get("displayName", "Home"):
        teamA_stats, teamB_stats = home_stats, away_stats
        teamA_name = home.get("team", {}).get("displayName", "Home")
        teamB_name = away.get("team", {}).get("displayName", "Away")
    else:
        teamA_stats, teamB_stats = away_stats, home_stats
        teamA_name = away.get("team", {}).get("displayName", "Away")
        teamB_name = home.get("team", {}).get("displayName", "Home")

    res = compute_matchup_and_player_prediction(teamA_stats, teamB_stats, player_profile,
                                                model_choice=model_choice, teamA_name=teamA_name, teamB_name=teamB_name)

    st.subheader("Results")
    st.write(f"{teamA_name} expected points: {res['teamA_expected']}")
    st.write(f"{teamB_name} expected points: {res['teamB_expected']}")
    st.write(f"Win probability — {teamA_name}: **{res['teamA_win_prob']}%**, {teamB_name}: **{res['teamB_win_prob']}%**")
    if stat_type == "fantasy_points":
        st.write(f"Predicted fantasy points for **{player_name}**: **{res.get('player_predicted_fantasy','N/A')}**")
    else:
        st.write(f"Predicted {stat_type} for **{player_name}**: **{res.get('player_predicted_stat_value','N/A')}**")

    if model_choice == "regression":
        fi = res.get("regression_info", {}).get("feature_importances", {})
        if fi:
            st.write("Regression model coefficients (feature importances):")
            fi_df = pd.DataFrame(list(fi.items()), columns=["feature", "coefficient"])
            st.dataframe(fi_df)
        else:
            st.write("Regression feature importances not available (sklearn missing or training failed).")

st.markdown("---")
st.caption("Workshop notes: API calls are cached. Regression uses a locally-created synthetic per-game CSV (demo_data/player_history.csv) so the training step is reproducible in the workshop without external downloads.")
