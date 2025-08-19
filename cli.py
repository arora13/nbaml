# cli.py
import re
import sys
import json
from pathlib import Path
from datetime import date
from dateutil import parser as dateparser

import pandas as pd
from joblib import load
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import leaguedashplayerstats

from features import build_dataset
from train import fetch_games

ART = Path("artifacts")

# ------------- parsing helpers -------------
SEASON_RE = re.compile(r'(\d{4})\s*[-/]\s*(\d{4})')
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

def fuzzy_team_id(name: str) -> int:
    name = name.strip().lower()
    all_teams = static_teams.get_teams()
    # direct full/partial
    for t in all_teams:
        if name in t["full_name"].lower():
            return t["id"]
    # abbreviation
    for t in all_teams:
        if name == t["abbreviation"].lower():
            return t["id"]
    # nicknames (last word, e.g., "lakers", "warriors")
    for t in all_teams:
        nickname = t["full_name"].split()[-1].lower()
        if name == nickname:
            return t["id"]
    raise ValueError(f"Team not found: '{name}'")

def parse_query(q: str):
    q = q.strip()
    # detect vs or @
    if " vs " in q.lower():
        sep = "vs"
    elif " @ " in q.lower():
        sep = "@"
    else:
        # try generic split on 'vs' or '@' without spaces
        if "vs" in q.lower():
            sep = "vs"
        elif "@" in q:
            sep = "@"
        else:
            raise ValueError("Could not find 'vs' or '@' between teams.")
    parts = re.split(r'\bvs\b|@', q, flags=re.IGNORECASE)
    if len(parts) < 2:
        raise ValueError("Could not split teams. Use 'TeamA vs TeamB' or 'TeamA @ TeamB'.")
    t1 = parts[0].strip()
    rest = parts[1].strip()

    # extract date if present
    date_match = DATE_RE.search(q)
    dt = None
    if date_match:
        dt = dateparser.parse(date_match.group(0)).date()

    # extract season if present (e.g., 2025-2026)
    season_match = SEASON_RE.search(q)
    season_start = None
    season_str = None
    if season_match:
        a, b = int(season_match.group(1)), int(season_match.group(2))
        # normalize direction (2025-2026 okay)
        season_start = a if a < b else b
        season_str = f"{season_start}-{str(season_start+1)[-2:]}"

    # clean opponent string by removing season phrase and date
    opp = SEASON_RE.sub("", rest)
    opp = DATE_RE.sub("", opp).strip()

    # decide home/away
    if sep == "vs":
        home_name, away_name = t1, opp
    else:  # '@' means first team is away at second team
        home_name, away_name = opp, t1

    return home_name, away_name, dt, season_start, season_str

# ------------- lineup helper -------------
def best_lineup_by_minutes(team_id: int, season_str: str, top_n: int = 5):
    """Use LeagueDashPlayerStats to grab per-game minutes and pick top 5."""
    try:
        ldp = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed="PerGame",
            team_id_nullable=team_id
        ).get_data_frames()[0]
        ldp = ldp.sort_values("MIN", ascending=False)
        names = ldp["PLAYER_NAME"].head(top_n).tolist()
        return names
    except Exception:
        return []

# ------------- prediction core -------------
def build_X_for_row(df_row: pd.DataFrame) -> pd.DataFrame:
    # Keep feature selection in sync with your trainers
    base = [
        "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
        "DOW","MONTH",
        "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF",
        "REST_R5_HOME","REST_R5_AWAY","B2B_RATE_R5_HOME","B2B_RATE_R5_AWAY",
        "REST_R5_DIFF","B2B_RATE_R5_DIFF"
    ]
    diffs = [c for c in df_row.columns if (
        c.endswith("_DIFF_R10") or c.endswith("_DIFF_R30") or c.endswith("_DIFF_SDT")
    )]
    return df_row[base + diffs].copy()

def load_best_model():
    # prefer HGB if present, else LR
    hgb_path = ART / "hgb_model.joblib"
    if hgb_path.exists():
        model = load(hgb_path)
        return ("hgb", model, None)  # no scaler
    # fall back to LR
    lr = ART / "model.joblib"
    sc = ART / "scaler.joblib"
    if lr.exists() and sc.exists():
        return ("lr", load(lr), load(sc))
    raise RuntimeError("No trained model found. Run `python train_hgb.py` or `python train.py` first.")

def predict_game(home_name: str, away_name: str, dt: date | None, season_start: int | None):
    hid = fuzzy_team_id(home_name)
    aid = fuzzy_team_id(away_name)

    # Decide season range to fetch
    if dt:
        history_end = dt.year
        seasons_text = f"{history_end-1}-{history_end}"
    elif season_start:
        history_end = season_start + 1  # season spans start->start+1
        seasons_text = f"{season_start}-{season_start+1}"
    else:
        # default: current year
        history_end = date.today().year
        seasons_text = f"{history_end-1}-{history_end}"

    # Fetch and build dataset
    team_rows = fetch_games(history_end-9, history_end)  # last ~10 seasons for context
    ds = build_dataset(team_rows)

    # If explicit date given → try to find that game
    row = pd.DataFrame()
    if dt:
        day_df = ds[ds["GAME_DATE"].dt.date == dt]
        row = day_df[(day_df["HOME_ID"] == hid) & (day_df["AWAY_ID"] == aid)]

    # If season given without date → look for first meeting of that season
    if row.empty and season_start:
        season_tag = str(season_start)
        season_df = ds[ds["SEASON_ID"].astype(str).str.contains(season_tag)]
        cand = season_df[(season_df["HOME_ID"] == hid) & (season_df["AWAY_ID"] == aid)]
        if cand.empty:
            # try reversed venue (maybe you said 'vs' but first was away)
            cand = season_df[(season_df["HOME_ID"] == aid) & (season_df["AWAY_ID"] == hid)]
            if not cand.empty:
                # flip venue if needed
                row = cand.sort_values("GAME_DATE").head(1).copy()
                # if reversed, swap so prediction reflects requested home/away
                row[["HOME_ID","AWAY_ID"]] = row[["AWAY_ID","HOME_ID"]].values
            else:
                # No scheduled/recorded game found → fallback: use latest prior season SDT info
                # pick the most recent game for each team to synthesize a neutral matchup
                last_home = ds[ds["HOME_ID"] == hid].tail(1)
                last_away = ds[ds["AWAY_ID"] == aid].tail(1)
                if not last_home.empty and not last_away.empty:
                    # Use the later date as anchor and merge features by manual diff
                    base_cols = list(set(ds.columns) - {"HOME_WIN"})
                    row = last_home[base_cols].copy()
        else:
            row = cand.sort_values("GAME_DATE").head(1).copy()

    if row.empty:
        raise ValueError("Could not find a game row to score (date/season not available yet). Try a specific date from a completed season.")

    # Build feature vector
    X = build_X_for_row(row)

    # Load model and predict
    kind, model, scaler = load_best_model()
    if kind == "lr":
        Xs = scaler.transform(X)
        proba = float(model.predict_proba(Xs)[:, 1][0])
    else:
        proba = float(model.predict_proba(X)[:, 1][0])

    # Winner guess
    winner_id = hid if proba >= 0.5 else aid
    winner = next(t["full_name"] for t in static_teams.get_teams() if t["id"] == winner_id)

    # Best lineups via season stats (if season string derived)
    season_for_lineups = None
    if dt:
        season_for_lineups = f"{dt.year-1}-{str(dt.year)[-2:]}"
    elif season_start:
        season_for_lineups = f"{season_start}-{str(season_start+1)[-2:]}"
    else:
        season_for_lineups = f"{history_end-1}-{str(history_end)[-2:]}"

    home_lineup = best_lineup_by_minutes(hid, season_for_lineups)
    away_lineup = best_lineup_by_minutes(aid, season_for_lineups)

    out = {
        "parsed": {
            "home": home_name,
            "away": away_name,
            "date": dt.isoformat() if dt else None,
            "season": seasons_text
        },
        "prediction": {
            "home_team": next(t["full_name"] for t in static_teams.get_teams() if t["id"] == hid),
            "away_team": next(t["full_name"] for t in static_teams.get_teams() if t["id"] == aid),
            "home_win_prob": proba,
            "predicted_winner": winner
        },
        "suggested_lineups_by_minutes": {
            "home_top5": home_lineup,
            "away_top5": away_lineup
        }
    }
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py \"grizzlies vs warriors 2025-2026 season\"")
        sys.exit(1)
    query = sys.argv[1]
    try:
        home_name, away_name, dt, season_start, _ = parse_query(query)
        result = predict_game(home_name, away_name, dt, season_start)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
