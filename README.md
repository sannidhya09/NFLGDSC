NFLGDSC — NFL Prediction App

This project was created by Sannidhya Tiwari for a Google Developer Student Club (GDSC) workshop, where I combine pieces of my personal interests and professional experience. Football has always been a passion of mine, and the GDSC is an organization I deeply respect. This project brings together data science, Python programming, and UI/UX development to create a real-time NFL prediction app.

Overview

The NFLGDSC app predicts player performance and game outcomes for upcoming NFL games. It uses both heuristic methods and regression models to give accurate predictions, while ensuring position-aware logic so predictions make sense for each player's role.

Features include:

--Win Probability Prediction

  Calculates the expected points for each team and the probability of each team winning.

  Takes into account team averages, opponent defensive strength, and home/away effects.

--Player Performance Prediction

  Predicts a player's stat (passing yards, rushing yards, receiving yards) or fantasy points.

  Position-aware: ensures predictions are valid for the player’s role (e.g., offensive linemen will not have passing yards).

Uses historical averages, team performance, and regression model outputs.

--Interactive UI with Streamlit

  Red & black theme for a bold, sports-focused aesthetic.

  Player selection with dropdown menus.

  Beautiful cards to display key stats for teams and players.

  Graphs and charts for team comparisons and predictions.

  Workshop-friendly: shows how data flows from API → processing → prediction → UI.

--Caching and Robustness

  API responses are cached to speed up repeated demos.

  Handles missing or incomplete data gracefully.