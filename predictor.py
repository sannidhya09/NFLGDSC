# predictor.py
"""
Small, clear prediction heuristics for the workshop.
- team win probability based on averaged offense/defense
- player stat prediction based on position-default averages and usage adjustments
"""

import math
from typing import Dict, Any

SCORE_SD = 13.0


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def team_expected_points(team_avg_points_scored: float, opp_avg_points_allowed: float) -> float:
    return (team_avg_points_scored + opp_avg_points_allowed) / 2.0


def win_probability(team_exp: float, opp_exp: float, sd: float = SCORE_SD) -> float:
    diff = team_exp - opp_exp
    return logistic(diff / sd)


def predict_player_stat(player_avg: float,
                        team_factor: float = 1.0,
                        opp_def_factor: float = 1.0,
                        usage: float = 1.0) -> float:
    if opp_def_factor <= 0:
        opp_def_factor = 1.0
    return max(0.0, player_avg * team_factor * (1.0 / opp_def_factor) * usage)


def fantasy_points_from_stats(stat_map: Dict[str, float]) -> float:
    """
    Simple fantasy scoring:
      passing_yds: 1 / 25
      passing_tds: 4
      rushing_yds: 1 / 10
      rushing_tds: 6
      receiving_yds: 1 / 10
      receiving_tds: 6
      ints/fumbles: -2
    """
    pts = 0.0
    pts += stat_map.get("passing_yds", 0.0) / 25.0
    pts += stat_map.get("passing_tds", 0.0) * 4.0
    pts += stat_map.get("rushing_yds", 0.0) / 10.0
    pts += stat_map.get("rushing_tds", 0.0) * 6.0
    pts += stat_map.get("receiving_yds", 0.0) / 10.0
    pts += stat_map.get("receiving_tds", 0.0) * 6.0
    pts += stat_map.get("fumbles", 0.0) * -2.0
    pts += stat_map.get("interceptions", 0.0) * -2.0
    return pts


# reasonable defaults by position (yards per game)
POSITION_DEFAULTS = {
    "QB": {"passing_yds": 220.0},
    "RB": {"rushing_yds": 55.0},
    "WR": {"receiving_yds": 60.0},
    "TE": {"receiving_yds": 40.0},
    "UNK": {"receiving_yds": 8.0}
}


def get_default_baseline(position_abbrev: str, stat_type: str) -> float:
    """
    Return a default baseline (yards per game) for a player of given position
    and the chosen stat_type.
    """
    pos = (position_abbrev or "").upper()
    if pos not in POSITION_DEFAULTS:
        pos = "UNK"
    defaults = POSITION_DEFAULTS[pos]
    # stat_type may be one of 'passing_yds','rushing_yds','receiving_yds','fantasy_points'
    if stat_type == "fantasy_points":
        # map to receiving_yds by default for fantasy conversion fallback
        stat_key = list(defaults.keys())[0]
    else:
        stat_key = stat_type
        if stat_key not in defaults:
            stat_key = list(defaults.keys())[0]
    return float(defaults.get(stat_key, 0.0))


def compute_matchup_and_player_prediction(teamA_stats: Dict[str, Any],
                                          teamB_stats: Dict[str, Any],
                                          player_profile: Dict[str, Any],
                                          metric: str = "yards") -> Dict[str, Any]:
    """
    player_profile: { 'avg_stat': float, 'stat_type': 'rushing_yds', 'usage': 1.0, 'position': 'RB' }
    metric: 'fantasy' or 'yards'
    Returns expected team points, win percentages, and predicted player stat/fantasy.
    """
    a_exp = team_expected_points(teamA_stats.get("avg_points_scored", 20.0),
                                 teamB_stats.get("avg_points_allowed", 23.0))
    b_exp = team_expected_points(teamB_stats.get("avg_points_scored", 20.0),
                                 teamA_stats.get("avg_points_allowed", 23.0))
    prob_a = win_probability(a_exp, b_exp)
    prob_b = 1.0 - prob_a

    team_factor = (teamA_stats.get("avg_points_scored", 20.0) / 20.0)
    opp_def_factor = teamB_stats.get("defense_strength", 1.0) or 1.0

    player_avg = player_profile.get("avg_stat", 50.0)
    usage = player_profile.get("usage", 1.0)

    predicted_raw = predict_player_stat(player_avg, team_factor, opp_def_factor, usage)

    result = {
        "teamA_expected": round(a_exp, 2),
        "teamB_expected": round(b_exp, 2),
        "teamA_win_prob": round(prob_a * 100, 1),
        "teamB_win_prob": round(prob_b * 100, 1),
        "player_predicted_stat_value": round(predicted_raw, 2)
    }

    if metric == "fantasy":
        # treat predicted_raw as yards of stat_type and compute fantasy points
        stat_type = player_profile.get("stat_type", "receiving_yds")
        stat_map = {}
        if stat_type == "passing_yds":
            stat_map["passing_yds"] = predicted_raw
        elif stat_type == "rushing_yds":
            stat_map["rushing_yds"] = predicted_raw
        else:
            stat_map["receiving_yds"] = predicted_raw
        result["player_predicted_fantasy"] = round(fantasy_points_from_stats(stat_map), 2)

    return result
