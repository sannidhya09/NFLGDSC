# app.py
import streamlit as st
import datetime
import pandas as pd
from typing import List, Dict

from data_fetcher import get_scoreboard_for_date, get_team_roster
from predictor import (compute_matchup_and_player_prediction,
                       get_default_baseline)

st.set_page_config(page_title="NFL Predictor", layout="wide")
st.title("🏈 Simple NFL Predictor — Workshop Edition (Final)")

# ---------- Helpers ----------
@st.cache_data(ttl=60)
def fetch_scoreboard(d: datetime.date) -> Dict:
    return get_scoreboard_for_date(d)


@st.cache_data(ttl=300)
def fetch_roster_cached(team_id: str) -> List[Dict]:
    # team_id may be string or int
    try:
        return get_team_roster(int(team_id))
    except Exception:
        return []

def tidy_record_summary(record_list):
    """
    record_list is often a list of dicts like [{'name':'overall','summary':'2-1'},...]
    Prefer the overall/total record; otherwise join summaries.
    """
    try:
        if not isinstance(record_list, list):
            return None
        for r in record_list:
            if (r.get("name") and r.get("name").lower() == "overall") or (r.get("type") and r.get("type").lower() == "total"):
                return r.get("summary") or r.get("displayValue") or str(r)
        # fallback join
        return ", ".join(str(r.get("summary") or r.get("displayValue") or "") for r in record_list if isinstance(r, dict))
    except Exception:
        return None

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


# ---------- Main UI ----------
date = st.date_input("Pick a date", value=datetime.date.today())
with st.spinner("Loading scoreboard..."):
    try:
        scoreboard = fetch_scoreboard(date)
        events = scoreboard.get("events", []) if scoreboard else []
    except Exception as e:
        st.error(f"Scoreboard fetch failed: {e}")
        st.stop()

# fallback demo
if not events:
    st.info("No live games found for that date — using demo matchup.")
    events = [{
        "id": "demo123",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"id": "1", "displayName": "Dallas Cowboys"}, "records": [{"summary":"2-1"}]},
                {"homeAway": "away", "team": {"id": "2", "displayName": "New York Giants"}, "records": [{"summary":"1-2"}]}
            ]
        }]
    }]

# Build game list
game_labels = []
parsed_games = []
for ev in events:
    comp = ev.get("competitions", [{}])[0]
    comps = comp.get("competitors", [])
    if not comps or len(comps) < 2:
        continue
    # attempt to find home/away
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

# Team headers and tidy record
col1, col2 = st.columns(2)
with col1:
    st.header(home.get("team", {}).get("displayName", "Home"))
    rec_txt = tidy_record_summary(home.get("records", []))
    if rec_txt:
        st.write("Record:", rec_txt)
with col2:
    st.header(away.get("team", {}).get("displayName", "Away"))
    rec_txt2 = tidy_record_summary(away.get("records", []))
    if rec_txt2:
        st.write("Record:", rec_txt2)

# Fetch rosters (cached)
home_id = home.get("team", {}).get("id") or home.get("team", {}).get("uid") or "1"
away_id = away.get("team", {}).get("id") or away.get("team", {}).get("uid") or "2"

with st.spinner("Fetching rosters (may take a moment)..."):
    home_players = fetch_roster_cached(str(home_id))
    away_players = fetch_roster_cached(str(away_id))

home_df = roster_to_df(home_players)
away_df = roster_to_df(away_players)

st.write("### Home roster (sample)")
st.dataframe(home_df.head(40))
st.write("### Away roster (sample)")
st.dataframe(away_df.head(40))

# Player selection: show name + position
team_choice = st.radio("Pick which team's player you'd like to predict for",
                       [home.get("team", {}).get("displayName", "Home"),
                        away.get("team", {}).get("displayName", "Away")])

players_df = home_df if team_choice == home.get("team", {}).get("displayName", "Home") else away_df
player_options = players_df.apply(lambda r: f"{r['name']} — {r['position'] or 'UNK'}|{r['id']}", axis=1).tolist()
# Format items for nice selectbox labels and parse back
labels = [opt.split("|")[0] for opt in player_options]
if not labels:
    labels = ["Demo Player"]
    player_options = ["Demo Player|0"]

player_choice_label = st.selectbox("Pick a player", labels)
selected_index = labels.index(player_choice_label)
selected_player_token = player_options[selected_index]
player_name, player_pos, player_id = (None, None, None)
if "|" in selected_player_token:
    name_part, id_part = selected_player_token.split("|", 1)
    player_name = name_part.split(" — ")[0].strip()
    player_pos = name_part.split(" — ")[1].strip() if " — " in name_part else "UNK"
    player_id = id_part
else:
    # fallback
    player_name = player_choice_label
    player_pos = "UNK"
    player_id = "0"

# Stat type selection (user chooses the metric category only)
stat_type = st.selectbox("Which stat do you want predicted for this player?",
                         ["passing_yds", "rushing_yds", "receiving_yds", "fantasy_points"])

# Automatically compute baseline for selected player:
#  - First we attempt to use any roster-provided stat we might have (rare)
#  - Otherwise fallback to position defaults from predictor.get_default_baseline
from predictor import get_default_baseline
baseline_est = None

# Attempt 1: if roster data included season averages (rare), look for them
# (our normalized roster doesn't include season stats, so skip this step - kept for future)
# Attempt 2: fallback to sensible defaults by position
baseline_est = get_default_baseline(player_pos or "UNK", stat_type)

st.markdown(f"**Baseline (auto)** — {player_name} — *{player_pos}* — estimated `{stat_type}` = **{baseline_est}** (yards/game or converted to fantasy if selected).")

# Advanced override (hidden by default)
with st.expander("Advanced: override baseline or usage (optional)"):
    override_baseline = st.number_input("Override baseline (yards per game)", value=float(baseline_est))
    usage = st.slider("Usage multiplier (0.2 low, 1 typical, 1.5 high)", 0.2, 2.0, 1.0, 0.05)
else_usage = 1.0
# If expander not opened, use default usage 1.0
try:
    # if override_baseline is defined (user opened expander), use it; otherwise fallback to baseline_est
    baseline_val = float(override_baseline)
    usage_val = float(usage)
except Exception:
    baseline_val = baseline_est
    usage_val = 1.0

# Team-level simple stats (we keep fixed demo values to keep app stable)
home_stats = {"avg_points_scored": 23, "avg_points_allowed": 21, "defense_strength": 1.0}
away_stats = {"avg_points_scored": 20, "avg_points_allowed": 24, "defense_strength": 1.0}

# Compute predictions button
if st.button("Compute predictions"):
    # Build player_profile
    player_profile = {
        "avg_stat": baseline_val,
        "stat_type": stat_type,
        "usage": usage_val,
        "position": player_pos
    }
    # Determine team A = selected player's team, team B = opponent
    if team_choice == home.get("team", {}).get("displayName", "Home"):
        teamA_stats, teamB_stats = home_stats, away_stats
    else:
        teamA_stats, teamB_stats = away_stats, home_stats

    metric = "fantasy" if stat_type == "fantasy_points" else "yards"
    res = compute_matchup_and_player_prediction(teamA_stats, teamB_stats, player_profile, metric=metric)

    st.subheader("Results")
    # Show clear labels with team names
    st.write(f"{team_choice} (Team A) expected points: {res['teamA_expected']}")
    # opponent name
    opponent_name = away.get("team", {}).get("displayName") if team_choice == home.get("team", {}).get("displayName") else home.get("team", {}).get("displayName")
    st.write(f"{opponent_name} (Team B) expected points: {res['teamB_expected']}")
    st.write(f"Win probability — {team_choice}: **{res['teamA_win_prob']}%**, {opponent_name}: **{res['teamB_win_prob']}%**")

    if metric == "fantasy":
        st.write(f"Predicted fantasy points for **{player_name}**: **{res.get('player_predicted_fantasy','N/A')}**")
        st.write(f"Predicted underlying stat ({stat_type}): **{res.get('player_predicted_stat_value','N/A')}**")
    else:
        st.write(f"Predicted {stat_type} for **{player_name}**: **{res.get('player_predicted_stat_value','N/A')}**")

st.caption("Notes: If live ESPN endpoints don't provide player season averages, the app uses position-based defaults. For production use, plug in a historical per-game dataset (nfl_data_py, nflfastR, or a paid sports data API) to compute accurate baselines.")
