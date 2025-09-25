# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from PIL import Image
import io
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from data_fetcher import fetch_scoreboard, fetch_roster, fetch_team_details
from predictor import (
    compute_matchup_and_player_prediction,
    get_valid_stats_for_position,
    get_default_baseline,
    ensure_and_train_demo_model,
)

# Page config
st.set_page_config(page_title="NFL Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Styles: red & black theme ---
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(180deg, #080808 0%, #1a0b0b 100%);
        color: #ffffff;
    }
    /* header */
    .header {
        background: linear-gradient(90deg, #b30000, #000000);
        padding: 18px;
        border-radius: 8px;
        color: white;
    }
    .team-card {
        background: rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 12px;
        border-radius: 10px;
    }
    .player-card {
        background: linear-gradient(180deg, rgba(20,0,0,0.6), rgba(0,0,0,0.6));
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .muted { color: #bbbbbb; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="header"><h1>🏈 NFL Predictor </h1><p class="muted">Live demo (ESPN) · Position-aware predictions · Regression & heuristic models</p></div>', unsafe_allow_html=True)



# Load scoreboard (cached)
events = fetch_scoreboard(pick_date.strftime("%Y%m%d"))
if not events:
    st.warning("No games found for that date — try another date or use demo matchups below.")
    # provide a simple demo matchup fallback
    events = [{
        "id": "demo",
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"id": "1", "displayName": "Pittsburgh Steelers", "logo": ""}, "records":[{"summary":"2-1"}]},
                {"homeAway": "away", "team": {"id": "2", "displayName": "Minnesota Vikings", "logo": ""}, "records":[{"summary":"2-1"}]}
            ]
        }]
    }]

# Build game list
game_labels = []
parsed = []
for ev in events:
    comp = ev.get("competitions", [{}])[0]
    comps = comp.get("competitors", [])
    if len(comps) < 2:
        continue
    home = next((c for c in comps if c.get("homeAway")=="home"), comps[0])
    away = next((c for c in comps if c.get("homeAway")=="away"), comps[1])
    label = f"{away['team']['displayName']} @ {home['team']['displayName']}"
    game_labels.append(label)
    parsed.append((home, away, ev))

st.sidebar.selectbox("Choose a game", game_labels, key="game_choice")
sel_idx = game_labels.index(st.session_state.get("game_choice", game_labels[0])) if game_labels else 0
home, away, sel_event = parsed[sel_idx]

# Helper: get team logo safely
def safe_logo(team_json):
    # endpoint sometimes contains 'logo' or 'links' -> skip if not present
    t = team_json.get("team", team_json) if isinstance(team_json, dict) else team_json
    return t.get("logo") or t.get("logos", [{}])[0].get("href") if isinstance(t, dict) else None

home_logo = safe_logo(home) or None
away_logo = safe_logo(away) or None

# Top-level matchup cards
col1, col2, col3 = st.columns([3, 3, 2])
with col1:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    if home_logo:
        try:
            r = requests.get(home_logo, timeout=5)
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            st.image(img, width=140)
        except Exception:
            pass
    st.markdown(f"### {home.get('team',{}).get('displayName','Home')}")
    rec_home = home.get("records", [])
    if isinstance(rec_home, list) and len(rec_home)>0:
        overall = next((r for r in rec_home if (r.get("name","")).lower()=="overall" or (r.get("type","")).lower()=="total"), rec_home[0])
        st.write(f"**Record:** {overall.get('summary', str(rec_home))}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    if away_logo:
        try:
            r = requests.get(away_logo, timeout=5)
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            st.image(img, width=140)
        except Exception:
            pass
    st.markdown(f"### {away.get('team',{}).get('displayName','Away')}")
    rec_away = away.get("records", [])
    if isinstance(rec_away, list) and len(rec_away)>0:
        overall = next((r for r in rec_away if (r.get("name","")).lower()=="overall" or (r.get("type","")).lower()=="total"), rec_away[0])
        st.write(f"**Record:** {overall.get('summary', str(rec_away))}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="team-card">', unsafe_allow_html=True)
    st.markdown("### Match")
    game_status = sel_event.get("status", {}).get("type", {}).get("description", "Scheduled")
    st.write(f"**Status:** {game_status}")
    scheduled = sel_event.get("date", "")
    st.write(f"**Date:** {scheduled.split('T')[0] if scheduled else pick_date}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Roster fetch (cached)
home_team_id = str(home.get("team", {}).get("id") or home.get("team", {}).get("uid") or "1")
away_team_id = str(away.get("team", {}).get("id") or away.get("team", {}).get("uid") or "2")
home_roster_json = fetch_roster(home_team_id)
away_roster_json = fetch_roster(away_team_id)

# normalize roster into list of players (id, name, position, maybe stats)
def normalize_roster(roster_json):
    players = []
    if not roster_json:
        return players
    athletes = roster_json.get("athletes") or roster_json.get("athletes", [])
    # ESPN structure: athletes is a list of groups; each group has items list
    for group in athletes if isinstance(athletes, list) else []:
        items = group.get("items") if isinstance(group, dict) else None
        if isinstance(items, list):
            for item in items:
                # item may include athlete/player info or be the athlete
                athlete = item.get("athlete") if isinstance(item, dict) and item.get("athlete") else item
                if isinstance(athlete, dict):
                    pid = athlete.get("id") or athlete.get("guid") or athlete.get("uid") or ""
                    name = athlete.get("displayName") or athlete.get("fullName") or athlete.get("shortName") or ""
                    pos = (item.get("position") or athlete.get("position") or {}).get("abbreviation") if (isinstance(item, dict) or isinstance(athlete, dict)) else ""
                    players.append({"id": str(pid), "name": name, "position": (pos or "").upper(), "athlete": athlete})
    return players

home_players = normalize_roster(home_roster_json)
away_players = normalize_roster(away_roster_json)

# Player pick UI
left, right = st.columns(2)
with left:
    st.subheader("Select Player (Home)")
    if home_players:
        home_select = st.selectbox("Home player", [f"{p['name']} — {p['position']}" for p in home_players], key="home_player")
    else:
        home_select = st.selectbox("Home player", ["Demo Player — QB"], key="home_player")
with right:
    st.subheader("Select Player (Away)")
    if away_players:
        away_select = st.selectbox("Away player", [f"{p['name']} — {p['position']}" for p in away_players], key="away_player")
    else:
        away_select = st.selectbox("Away player", ["Demo Player — WR"], key="away_player")

# Which player will we predict?
player_choice_label = st.radio("Which player to predict?", ["Home: " + home_select, "Away: " + away_select])
# extract selection
sel_player = None
sel_side = "home" if player_choice_label.startswith("Home:") else "away"
if sel_side == "home":
    sel_idx = [f"{p['name']} — {p['position']}" for p in home_players].index(home_select) if home_players else 0
    sel_player = home_players[sel_idx] if home_players else {"id":"0","name":"Demo Player","position":"QB"}
    sel_team_stats = {"avg_points_scored":23.0,"avg_points_allowed":21.0,"defense_strength":1.0,"team_factor":1.05}
    opp_team_stats = {"avg_points_scored":20.0,"avg_points_allowed":24.0,"defense_strength":1.0,"team_factor":0.95}
    sel_team_name = home.get("team",{}).get("displayName","Home")
    opp_team_name = away.get("team",{}).get("displayName","Away")
else:
    sel_idx = [f"{p['name']} — {p['position']}" for p in away_players].index(away_select) if away_players else 0
    sel_player = away_players[sel_idx] if away_players else {"id":"0","name":"Demo Player","position":"WR"}
    sel_team_stats = {"avg_points_scored":20.0,"avg_points_allowed":24.0,"defense_strength":1.0,"team_factor":0.95}
    opp_team_stats = {"avg_points_scored":23.0,"avg_points_allowed":21.0,"defense_strength":1.0,"team_factor":1.05}
    sel_team_name = away.get("team",{}).get("displayName","Away")
    opp_team_name = home.get("team",{}).get("displayName","Home")

# Get valid stats for position and show dropdown accordingly
position = sel_player.get("position","UNK")
valid_stats = get_valid_stats_for_position(position)
st.markdown("---")
st.subheader(f"Player card — {sel_player.get('name')} ({position})")

# player image attempt
player_photo_url = None
ath = sel_player.get("athlete") if sel_player.get("athlete") else {}
if isinstance(ath, dict):
    # ESPN sometimes has photos under 'headshot' or 'images' etc -- attempt to find a url
    photo = ath.get("headshot") or ath.get("photos") or ath.get("images") or {}
    if isinstance(photo, dict):
        player_photo_url = photo.get("href") or photo.get("url")
    elif isinstance(photo, list) and photo:
        # take first
        p0 = photo[0]
        player_photo_url = p0.get("href") or p0.get("url") if isinstance(p0, dict) else None

# big player card left
pc1, pc2 = st.columns([2,3])
with pc1:
    st.markdown('<div class="player-card">', unsafe_allow_html=True)
    if player_photo_url:
        try:
            r = requests.get(player_photo_url, timeout=4)
            img = Image.open(io.BytesIO(r.content)).convert("RGBA")
            st.image(img, width=160)
        except Exception:
            pass
    st.markdown(f"### {sel_player.get('name')}")
    st.markdown(f"**Position:** {position}")
    st.markdown(f"**Team:** {sel_team_name}")
    st.markdown("</div>", unsafe_allow_html=True)
with pc2:
    st.markdown('<div class="player-card">', unsafe_allow_html=True)
    baseline = get_default_baseline(position, valid_stats[0]) if valid_stats else 0.0
    st.markdown(f"**Baseline ({'yards/game' if valid_stats else 'N/A'}):** {baseline}")
    st.markdown(f"**Model:** {model_choice.title()} (toggle in sidebar)")
    if not valid_stats:
        st.error(f"No predictive stats available for position {position}. Predictions will be 0.0.")
    st.markdown("</div>", unsafe_allow_html=True)

# choose stat if available
stat_type = None
if valid_stats:
    stat_type = st.selectbox("Choose stat to predict", valid_stats, index=0)
    st.caption("Only position-relevant stats are shown — this prevents nonsense predictions (e.g., passing yards for an OT).")

# Predict button
if stat_type:
    if model_choice == "regression":
        ensure_and_train_demo_model()
    if st.button("Compute predictions"):
        res = compute_matchup_and_player_prediction(
            sel_team_stats, opp_team_stats,
            {"fullName": sel_player.get("name"), "position": position, "avg_stat": baseline},
            stat_type=stat_type, model_choice=model_choice,
            teamA_name=sel_team_name, teamB_name=opp_team_name
        )

        # Top matchup summary cards & charts
        leftcol, midcol, rightcol = st.columns([3,3,4])
        with leftcol:
            st.markdown('<div class="team-card">', unsafe_allow_html=True)
            st.markdown(f"#### {sel_team_name}")
            st.metric("Expected points", f"{res['teamA_expected']}")
            st.markdown(f"Win probability: **{res['teamA_win_prob']}%**")
            st.markdown("</div>", unsafe_allow_html=True)
        with midcol:
            st.markdown('<div class="team-card">', unsafe_allow_html=True)
            st.markdown(f"#### {opp_team_name}")
            st.metric("Expected points", f"{res['teamB_expected']}")
            st.markdown(f"Win probability: **{res['teamB_win_prob']}%**")
            st.markdown("</div>", unsafe_allow_html=True)
        with rightcol:
            # pie chart of win probabilities
            fig = go.Figure(data=[go.Pie(labels=[sel_team_name, opp_team_name],
                                         values=[res['teamA_win_prob'], res['teamB_win_prob']],
                                         hole=0.4)])
            fig.update_traces(textinfo='label+percent', marker=dict(colors=['#b30000','#000000']))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Player prediction")
        if res.get("message"):
            st.error(res["message"])
            st.info("If you expected a different metric, choose a player with a matching position (QB/RB/WR/TE).")
        else:
            st.success(f"Predicted {stat_type} for **{res['player_name']} ({res['player_position']})**: **{res['player_predicted_stat_value']}**")
            st.write(f"Predicted fantasy points (approx): **{res['player_predicted_fantasy']}**")

            # show coefficients if regression
            coeffs = res.get("coeffs") or {}
            if coeffs:
                st.markdown("**Regression coefficients (feature importances)**")
                coeff_df = pd.DataFrame(list(coeffs.items()), columns=["feature","coef"])
                st.dataframe(coeff_df.style.format({"coef":"{:.3f}"}), height=200)

            # player-stat comparison chart: baseline vs predicted
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Baseline (yards/game)", x=[sel_player.get("name")], y=[baseline], marker_color='#b30000'))
            fig2.add_trace(go.Bar(name="Predicted", x=[sel_player.get("name")], y=[res['player_predicted_stat_value']], marker_color='#222222'))
            fig2.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Cards, charts and position-aware predictions. This is an educational demo — not betting advice.")
