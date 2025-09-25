# predictor.py
"""
Predictions library:
- ELO quick implementation
- Heuristic predictor (multiplicative baseline)
- Regression demo model using scikit-learn trained on synthetic/demo data
- Demo CSV creation for per-game player history so the regression can be illustrated
"""

import os
import math
import random
import csv
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Try importing sklearn; if missing, regression features will be disabled gracefully.
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- ELO ----------
ELO_BASE = 1500.0
elo_ratings: Dict[str, float] = {}

def get_elo(team: str) -> float:
    return elo_ratings.get(team, ELO_BASE)

def set_elo(team: str, rating: float):
    elo_ratings[team] = rating

def update_elo_after_match(winner: str, loser: str, margin: float = 1.0, k: float = 20.0):
    Ra = get_elo(winner)
    Rb = get_elo(loser)
    Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400.0))
    Eb = 1.0 - Ea
    mf = math.log(abs(margin) + 1) * (2.2 / ((Ra - Rb) * 0.001 + 2.2)) if margin else 1.0
    Ra_new = Ra + k * (1 - Ea) * mf
    Rb_new = Rb + k * (0 - Eb) * mf
    set_elo(winner, Ra_new)
    set_elo(loser, Rb_new)

def elo_predict(teamA: str, teamB: str) -> (float, float):
    Ra = get_elo(teamA)
    Rb = get_elo(teamB)
    probA = 1 / (1 + 10 ** ((Rb - Ra) / 400.0))
    return probA, 1 - probA

# ---------- Heuristic predictor ----------
SCORE_SD = 13.0

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def team_expected_points(team_avg_points_scored: float, opp_avg_points_allowed: float) -> float:
    return (team_avg_points_scored + opp_avg_points_allowed) / 2.0

def win_probability_by_expected(a_exp: float, b_exp: float, sd: float = SCORE_SD) -> float:
    return logistic((a_exp - b_exp) / sd)

def predict_player_stat_heuristic(player_avg: float, team_factor: float = 1.0,
                                  opp_def_factor: float = 1.0, usage: float = 1.0) -> float:
    if opp_def_factor <= 0:
        opp_def_factor = 1.0
    return max(0.0, player_avg * team_factor * (1.0 / opp_def_factor) * usage)

# ---------- Position defaults & demo CSV ----------
POSITION_DEFAULTS = {
    "QB": {"passing_yds": 220.0},
    "RB": {"rushing_yds": 55.0},
    "WR": {"receiving_yds": 60.0},
    "TE": {"receiving_yds": 40.0},
    "UNK": {"receiving_yds": 8.0}
}

def get_default_baseline(position_abbrev: str, stat_type: str) -> float:
    pos = (position_abbrev or "UNK").upper()
    defaults = POSITION_DEFAULTS.get(pos, POSITION_DEFAULTS["UNK"])
    if stat_type == "fantasy_points":
        key = next(iter(defaults.keys()))
    else:
        key = stat_type if stat_type in defaults else next(iter(defaults.keys()))
    return float(defaults.get(key, 0.0))

# Demo data file path
DEMO_DIR = "demo_data"
DEMO_PLAYER_HISTORY = os.path.join(DEMO_DIR, "player_history.csv")

def ensure_demo_player_history(force_recreate: bool = False) -> str:
    """
    Create a small synthetic per-game CSV used to demo regression training if the file doesn't exist
    or if it exists but is empty/corrupt. Returns the absolute path.
    """
    if not os.path.isdir(DEMO_DIR):
        os.makedirs(DEMO_DIR, exist_ok=True)

    path = os.path.abspath(DEMO_PLAYER_HISTORY)
    recreate = False
    if force_recreate:
        recreate = True
    elif not os.path.exists(path):
        recreate = True
    else:
        try:
            size = os.path.getsize(path)
            if size < 10:  # tiny file, probably invalid
                recreate = True
        except Exception:
            recreate = True

    if not recreate:
        return path

    # Build synthetic dataset for ~8 games x 8 players
    players = [
        ("Patrick Mahomes", "KC", "QB"),
        ("Derrick Henry", "TEN", "RB"),
        ("Cooper Kupp", "LAR", "WR"),
        ("Travis Kelce", "KC", "TE"),
        ("Joe Burrow", "CIN", "QB"),
        ("Jonathan Taylor", "IND", "RB"),
        ("Justin Jefferson", "MIN", "WR"),
        ("Darren Waller", "NYG", "TE"),
    ]
    rows = []
    for name, team, pos in players:
        baseline = get_default_baseline(pos, "passing_yds" if pos == "QB" else ("rushing_yds" if pos == "RB" else "receiving_yds"))
        for g in range(8):
            yards = max(0, int(random.gauss(baseline, max(1.0, baseline * 0.25))))
            rows.append((name, team, pos, f"2024-10-{g+1:02d}", yards))
    # write CSV robustly using pandas (safer than manual writer)
    try:
        df = pd.DataFrame(rows, columns=["player", "team", "position", "date", "game_yards"])
        df.to_csv(path, index=False)
    except Exception:
        # fallback: manual csv writer
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["player", "team", "position", "date", "game_yards"])
                for r in rows:
                    writer.writerow(r)
        except Exception:
            # Last-resort: create a tiny file with header + one row
            with open(path, "w", encoding="utf-8") as f:
                f.write("player,team,position,date,game_yards\nDemo Player,DM,UNK,2024-10-01,10\n")
    return path

# ---------- Regression model (demo) ----------
class DemoRegressionModel:
    def __init__(self):
        self.model = None
        self.coef_names = []
        self.is_trained = False
        if SKLEARN_AVAILABLE:
            self.model = LinearRegression()
        else:
            self.model = None

    def prepare_training_df(self, csv_path: str) -> pd.DataFrame:
        # Attempt to safely load CSV; if it fails, regenerate
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            csv_path = ensure_demo_player_history(force_recreate=True)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                # last fallback: create small in-memory dataframe
                df = pd.DataFrame({
                    "player": ["Demo Player"],
                    "position": ["UNK"],
                    "team": ["DM"],
                    "game_yards": [10]
                })
        # group and create features
        df_agg = df.groupby(["player", "position", "team"]).agg(
            player_avg=("game_yards", "mean"),
            games=("game_yards", "count")
        ).reset_index()
        df_agg["opp_def_factor"] = np.random.uniform(0.8, 1.2, size=len(df_agg))
        df_agg["team_factor"] = np.random.uniform(0.9, 1.2, size=len(df_agg))
        df_agg["y"] = (df_agg["player_avg"] * np.random.uniform(0.8, 1.2, size=len(df_agg))).round(1)
        return df_agg

    def train(self, csv_path: str) -> bool:
        """
        Train the demo model. Returns True if trained, False otherwise.
        """
        if not SKLEARN_AVAILABLE:
            self.is_trained = False
            return False
        try:
            df = self.prepare_training_df(csv_path)
            if df.empty:
                self.is_trained = False
                return False
            X = df[["player_avg", "team_factor", "opp_def_factor"]].values
            y = df["y"].values
            self.model.fit(X, y)
            self.coef_names = ["player_avg", "team_factor", "opp_def_factor"]
            self.is_trained = True
            return True
        except Exception:
            self.is_trained = False
            return False

    def predict(self, player_avg: float, team_factor: float, opp_def_factor: float) -> float:
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return predict_player_stat_heuristic(player_avg, team_factor, opp_def_factor, usage=1.0)
        X = np.array([[player_avg, team_factor, opp_def_factor]])
        y_hat = self.model.predict(X)[0]
        return float(max(0.0, y_hat))

    def feature_importances(self) -> Dict[str, float]:
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return {}
        return {name: float(coef) for name, coef in zip(self.coef_names, self.model.coef_.tolist())}

# single demo model instance
demo_reg_model = DemoRegressionModel()

def ensure_and_train_demo_model() -> DemoRegressionModel:
    """
    Ensure demo CSV exists and attempt to train the demo regression model.
    This function never raises: it will return the model instance (trained or not).
    """
    csv_path = ensure_demo_player_history()
    try:
        demo_reg_model.train(csv_path)
    except Exception:
        # swallow and leave demo_reg_model.is_trained False
        demo_reg_model.is_trained = False
    return demo_reg_model

# ---------- Master orchestrator ----------
def compute_matchup_and_player_prediction(teamA_stats: Dict[str, Any],
                                          teamB_stats: Dict[str, Any],
                                          player_profile: Dict[str, Any],
                                          model_choice: str = "heuristic",
                                          teamA_name: str = "TeamA",
                                          teamB_name: str = "TeamB") -> Dict[str, Any]:
    try:
        a_exp = team_expected_points(teamA_stats.get("avg_points_scored", 20.0),
                                     teamB_stats.get("avg_points_allowed", 23.0))
        b_exp = team_expected_points(teamB_stats.get("avg_points_scored", 20.0),
                                     teamA_stats.get("avg_points_allowed", 23.0))
    except Exception:
        a_exp, b_exp = 20.0, 20.0

    if model_choice == "elo":
        if teamA_name not in elo_ratings:
            set_elo(teamA_name, ELO_BASE + (a_exp - 21.0) * 10)
        if teamB_name not in elo_ratings:
            set_elo(teamB_name, ELO_BASE + (b_exp - 21.0) * 10)
        pA, pB = elo_predict(teamA_name, teamB_name)
        teamA_win_prob = pA
        teamB_win_prob = pB
    else:
        teamA_win_prob = win_probability_by_expected(a_exp, b_exp)
        teamB_win_prob = 1.0 - teamA_win_prob

    position = (player_profile.get("position") or "UNK").upper()
    stat_type = player_profile.get("stat_type", "receiving_yds")
    usage = float(player_profile.get("usage", 1.0))
    player_avg = float(player_profile.get("avg_stat", get_default_baseline(position, stat_type)))

    team_factor = (teamA_stats.get("avg_points_scored", 20.0) / 20.0)
    opp_def_factor = teamB_stats.get("defense_strength", 1.0) or 1.0

    reg_info = {}
    if model_choice == "regression":
        model = ensure_and_train_demo_model()
        pred_raw = model.predict(player_avg, team_factor, opp_def_factor) * usage
        reg_info["feature_importances"] = model.feature_importances()
    else:
        pred_raw = predict_player_stat_heuristic(player_avg, team_factor, opp_def_factor, usage)

    if stat_type == "passing_yds":
        player_pred_fantasy = pred_raw / 25.0
    elif stat_type == "rushing_yds":
        player_pred_fantasy = pred_raw / 10.0
    else:
        player_pred_fantasy = pred_raw / 10.0

    return {
        "teamA_expected": round(a_exp, 2),
        "teamB_expected": round(b_exp, 2),
        "teamA_win_prob": round(teamA_win_prob * 100, 1),
        "teamB_win_prob": round(teamB_win_prob * 100, 1),
        "player_predicted_stat_value": round(pred_raw, 2),
        "player_predicted_fantasy": round(player_pred_fantasy, 2),
        "regression_info": reg_info
    }
