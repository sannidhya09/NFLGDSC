# predictor.py
"""
Prediction library:
- position-aware stat mapping (prevents nonsense predictions)
- heuristic & regression demo model (safe)
- ELO-ish optional support (kept simple)
- generates/uses demo CSV for regression training so app is reproducible
"""

import os
import math
import random
import csv
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# try sklearn
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# Position -> valid stats
# -------------------------
POSITION_STAT_MAP = {
    "QB": ["passing_yds"],
    "RB": ["rushing_yds"],
    "WR": ["receiving_yds"],
    "TE": ["receiving_yds"],
    # Kickers, defensive positions etc. left out for simplicity — considered non-predictive here
}

NO_STAT_POSITIONS = {
    "OT", "OG", "C", "LT", "RT", "G", "T", "FB",
    "LB", "CB", "S", "DE", "DT", "NT", "DL", "DB", "K", "P"
}

def get_valid_stats_for_position(position: str) -> List[str]:
    pos = (position or "").upper()
    return POSITION_STAT_MAP.get(pos, [])

# -------------------------
# Baseline defaults by position and stat
# -------------------------
POSITION_DEFAULTS = {
    "QB": {"passing_yds": 220.0},
    "RB": {"rushing_yds": 55.0},
    "WR": {"receiving_yds": 60.0},
    "TE": {"receiving_yds": 40.0},
    "UNK": {"receiving_yds": 8.0}
}

def get_default_baseline(position: str, stat_type: str) -> float:
    pos = (position or "UNK").upper()
    defaults = POSITION_DEFAULTS.get(pos, POSITION_DEFAULTS["UNK"])
    if stat_type in defaults:
        return float(defaults[stat_type])
    # fallback to first available stat in defaults
    return float(next(iter(defaults.values())))

# -------------------------
# Small ELO-like helper (optional)
# -------------------------
ELO_BASE = 1500.0
elo_ratings: Dict[str, float] = {}

def get_elo(team: str) -> float:
    return elo_ratings.get(team, ELO_BASE)

def set_elo(team: str, rating: float):
    elo_ratings[team] = rating

def elo_predict(teamA: str, teamB: str) -> (float, float):
    Ra = get_elo(teamA)
    Rb = get_elo(teamB)
    probA = 1 / (1 + 10 ** ((Rb - Ra) / 400.0))
    return probA, 1 - probA

# -------------------------
# Heuristic expected points/win prob
# -------------------------
SCORE_SD = 13.0
def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def team_expected(team_avg: float, opp_allowed: float) -> float:
    return (team_avg + opp_allowed) / 2.0

def win_prob_by_expected(a_exp: float, b_exp: float) -> float:
    return logistic((a_exp - b_exp) / SCORE_SD)

# -------------------------
# Demo regression model & demo csv generation
# -------------------------
DEMO_DIR = "demo_data"
DEMO_CSV = os.path.join(DEMO_DIR, "player_history.csv")

def ensure_demo_csv(force=False) -> str:
    os.makedirs(DEMO_DIR, exist_ok=True)
    if not force and os.path.exists(DEMO_CSV) and os.path.getsize(DEMO_CSV) > 20:
        return DEMO_CSV
    # generate tiny synthetic dataset (players x games)
    players = [
        ("Demo QB", "QB", 220.0),
        ("Demo RB", "RB", 55.0),
        ("Demo WR", "WR", 60.0),
        ("Demo TE", "TE", 40.0),
    ]
    rows = []
    for name, pos, base in players:
        for g in range(8):
            yards = max(0, int(random.gauss(base, base * 0.25)))
            rows.append((name, pos, f"2024-10-{g+1:02d}", yards))
    # write csv
    try:
        with open(DEMO_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["player", "position", "date", "game_yards"])
            writer.writerows(rows)
    except Exception:
        # final fallback: write one line minimal
        with open(DEMO_CSV, "w", encoding="utf-8") as f:
            f.write("player,position,date,game_yards\nDemo Player,UNK,2024-10-01,10\n")
    return DEMO_CSV

class DemoRegressionModel:
    def __init__(self):
        self.model = None
        self.coef_names = ["player_avg", "team_factor", "opp_def_factor"]
        self.trained = False

    def train(self, csv_path=None):
        if not SKLEARN_AVAILABLE:
            self.trained = False
            return False
        csv_path = csv_path or ensure_demo_csv()
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            csv_path = ensure_demo_csv(force=True)
            df = pd.read_csv(csv_path)
        # aggregate to player averages
        agg = df.groupby(["player", "position"]).agg(player_avg=("game_yards", "mean")).reset_index()
        # synthetic team_factor & opp_def
        agg["team_factor"] = np.random.uniform(0.8, 1.2, size=len(agg))
        agg["opp_def_factor"] = np.random.uniform(0.8, 1.2, size=len(agg))
        # target simulated as slightly noisy player_avg
        agg["y"] = (agg["player_avg"] * np.random.uniform(0.8, 1.2, size=len(agg))).round(1)
        X = agg[["player_avg", "team_factor", "opp_def_factor"]].values
        y = agg["y"].values
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(X, y)
            self.model = lr
            self.trained = True
            return True
        except Exception:
            self.trained = False
            return False

    def predict(self, player_avg, team_factor, opp_def_factor):
        if not self.trained or self.model is None:
            return player_avg
        x = np.array([[player_avg, team_factor, opp_def_factor]])
        yhat = float(self.model.predict(x)[0])
        return max(0.0, yhat)

    def coefficients(self):
        if not self.trained or self.model is None:
            return {}
        return dict(zip(self.coef_names, list(self.model.coef_)))

# single instance
demo_model = DemoRegressionModel()

def ensure_and_train_demo_model():
    if not demo_model.trained:
        demo_model.train(ensure_demo_csv())
    return demo_model

# -------------------------
# Master prediction orchestrator (position-aware)
# -------------------------
def compute_matchup_and_player_prediction(teamA_stats: Dict[str, Any],
                                          teamB_stats: Dict[str, Any],
                                          player_profile: Dict[str, Any],
                                          stat_type: str = "receiving_yds",
                                          model_choice: str = "heuristic",
                                          teamA_name: str = "TeamA",
                                          teamB_name: str = "TeamB") -> Dict[str, Any]:
    """
    Returns:
      teamA_expected, teamB_expected, teamA_win_prob_pct, teamB_win_prob_pct,
      player_pred_stat (float), player_pred_fantasy (float), coeffs (dict), message (str or None)
    """
    # team expected points: use provided stats or fallbacks
    try:
        a_exp = team_expected(teamA_stats.get("avg_points_scored", 22.0), teamB_stats.get("avg_points_allowed", 23.0))
        b_exp = team_expected(teamB_stats.get("avg_points_scored", 21.0), teamA_stats.get("avg_points_allowed", 24.0))
    except Exception:
        a_exp, b_exp = 22.0, 21.0

    # use chosen win-prob method
    if model_choice == "elo":
        pA, pB = elo_predict(teamA_name, teamB_name)
    else:
        pA = win_prob_by_expected(a_exp, b_exp)
        pB = 1.0 - pA

        # player checks
        player_name = player_profile.get("fullName", player_profile.get("name", "Unknown"))
        position = player_profile.get("position", "").upper() or "UNK"
        valid_stats = get_valid_stats_for_position(position)
    if stat_type not in valid_stats:
        # Not a valid stat for this position: force 0.0 and descriptive message
        return {
            "teamA_expected": round(a_exp, 2),
            "teamB_expected": round(b_exp, 2),
            "teamA_win_prob": round(pA * 100, 1),
            "teamB_win_prob": round(pB * 100, 1),
            "player_name": player_name,
            "player_position": position,
            "player_predicted_stat_value": 0.0,
            "player_predicted_fantasy": 0.0,
            "coeffs": {},
            "message": f"No predictive {stat_type} available for {position} players (predicted = 0)."
        }

    # baseline (if player has historical avg, it should be provided, otherwise fallback)
    player_avg = float(player_profile.get("avg_stat", get_default_baseline(position, stat_type)))
    team_factor = teamA_stats.get("team_factor", 1.0)
    opp_def_factor = teamB_stats.get("defense_strength", 1.0)

    coeffs = {}
    if model_choice == "regression":
        model = ensure_and_train_demo_model()
        pred = model.predict(player_avg, team_factor, opp_def_factor)
        coeffs = model.coefficients()
    else:
        # simple heuristic: adjust baseline by team/opp multipliers
        pred = player_avg * team_factor * (1.0 / (opp_def_factor if opp_def_factor > 0 else 1.0))

    # fantasy conversion simple
    if stat_type == "passing_yds":
        fantasy = pred / 25.0
    else:
        # rushing/receiving
        fantasy = pred / 10.0

    return {
        "teamA_expected": round(a_exp, 2),
        "teamB_expected": round(b_exp, 2),
        "teamA_win_prob": round(pA * 100, 1),
        "teamB_win_prob": round(pB * 100, 1),
        "player_name": player_name,
        "player_position": position,
        "player_predicted_stat_value": round(pred, 2),
        "player_predicted_fantasy": round(fantasy, 2),
        "coeffs": coeffs,
        "message": None
    }
