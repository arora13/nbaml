# predict.py
import re
import sys
import json
from pathlib import Path
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import leaguedashplayerstats

from features import build_dataset  # your engineered features

ART = Path("artifacts")
DATA_CSV = Path("data/games.csv")

# ---------------- Parsing ----------------
SEASON_RE = re.compile(r'(\d{4})\s*[-/]\s*(\d{4})')
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

def parse_query(args: list) -> str:
    if len(args) == 1:
        return args[0]
    if len(args) >= 2:
        rest = " ".join(args[2:]).strip()
        return f"{args[0]} vs {args[1]}{(' ' + rest) if rest else ''}"
    raise ValueError("Usage: python predict.py \"GSW vs LAL 2026-10-24\" OR python predict.py GSW LAL [2026-10-24]")

def parse_query_text(q: str):
    q = q.strip()
    sep = None
    if re.search(r'\bvs\b', q, re.IGNORECASE): sep = 'vs'
    elif '@' in q: sep = '@'
    else:
        if 'vs' in q.lower(): sep = 'vs'
        elif '@' in q: sep = '@'
    if not sep:
        raise ValueError("Could not find 'vs' or '@' between teams.")

    parts = re.split(r'\bvs\b|@', q, flags=re.IGNORECASE)
    if len(parts) < 2:
        raise ValueError("Could not split teams.")

    left, right = parts[0].strip(), parts[1].strip()

    # optional date
    dt = None
    mdate = DATE_RE.search(q)
    if mdate:
        from dateutil import parser as dateparser
        dt = dateparser.parse(mdate.group(0)).date()

    # optional season like 2025-2026
    season_start = None
    season_str = None
    mseason = SEASON_RE.search(q)
    if mseason:
        a, b = int(mseason.group(1)), int(mseason.group(2))
        season_start = min(a, b)
        season_str = f"{season_start}-{str(season_start+1)[-2:]}"

    # strip date/season tokens from team strings
    clean_right = re.sub(SEASON_RE, '', re.sub(DATE_RE, '', right)).strip()

    # resolve home/away
    if sep == 'vs':
        home_name, away_name = left, clean_right
    else:  # '@' means left is away at right
        home_name, away_name = clean_right, left

    return home_name, away_name, dt, season_start, season_str

# ---------------- Data loading ----------------
def load_team_rows_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["TEAM_ID"] = pd.to_numeric(df["TEAM_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["GAME_DATE", "TEAM_ID", "MATCHUP", "WL"]).copy()
    df["TEAM_ID"] = df["TEAM_ID"].astype(int)
    keep = [
        "SEASON_ID","GAME_ID","GAME_DATE","MATCHUP","WL","TEAM_ID",
        "PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","TOV","STL","BLK","PF","PLUS_MINUS"
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep].sort_values("GAME_DATE").reset_index(drop=True)

# ---------------- Teams ----------------
def fuzzy_team_id(name: str) -> int:
    name = name.strip().lower()
    teams = static_teams.get_teams()
    for t in teams:
        if name in t["full_name"].lower():
            return t["id"]
    for t in teams:
        if name == t["abbreviation"].lower():
            return t["id"]
    for t in teams:
        if name == t["full_name"].split()[-1].lower():
            return t["id"]
    raise ValueError(f"Team not found: '{name}'")

def team_name(team_id: int) -> str:
    for t in static_teams.get_teams():
        if t["id"] == team_id:
            return t["full_name"]
    return str(team_id)

# ---------------- Build one-row feature frame ----------------
BASE_WANTED = [
    "HOME_ELO_PRE","AWAY_ELO_PRE","HOME_ELO_EXP","ELO_DIFF",
    "DOW","MONTH",
    "REST_HOME","REST_AWAY","B2B_HOME","B2B_AWAY","REST_DIFF","B2B_DIFF",
    "REST_R5_HOME","REST_R5_AWAY","B2B_RATE_R5_HOME","B2B_RATE_R5_AWAY",
    "REST_R5_DIFF","B2B_RATE_R5_DIFF"
]

def build_X_for_row(df_row: pd.DataFrame) -> pd.DataFrame:
    for col in BASE_WANTED:
        if col not in df_row.columns:
            df_row[col] = 0.0
    diff_cols = [c for c in df_row.columns
                 if c.endswith("_DIFF_R10") or c.endswith("_DIFF_R30") or c.endswith("_DIFF_SDT")]
    return df_row[BASE_WANTED + diff_cols].copy()

def _latest_team_game(ds: pd.DataFrame, team_id: int, before: Optional[date]):
    dss = ds
    if before is not None:
        dss = ds[ds["GAME_DATE"].dt.date <= before]
    subset = dss[(dss["HOME_ID"] == team_id) | (dss["AWAY_ID"] == team_id)]
    if subset.empty:
        return None
    return subset.sort_values("GAME_DATE").iloc[-1]

def _collect_side_features(src_row: pd.Series, team_id: int, target_side: str) -> dict:
    out = {}
    was_home = bool(src_row["HOME_ID"] == team_id)
    src_side = "HOME" if was_home else "AWAY"
    for col in src_row.index:
        if col.endswith(f"_{src_side}"):
            base = col[:-(len(src_side)+1)]
            out[f"{base}_{target_side}"] = src_row[col]
    out[f"{'HOME' if target_side=='HOME' else 'AWAY'}_ELO_PRE"] = (
        src_row["HOME_ELO_PRE"] if was_home else src_row["AWAY_ELO_PRE"]
    )
    return out

def synthesize_matchup_row(ds: pd.DataFrame, hid: int, aid: int, tgt_date: Optional[date]) -> pd.DataFrame:
    hrow = _latest_team_game(ds, hid, tgt_date)
    arow = _latest_team_game(ds, aid, tgt_date)
    if hrow is None or arow is None:
        raise ValueError("Not enough history to synthesize matchup.")

    out = {
        "HOME_ID": hid,
        "AWAY_ID": aid,
        "GAME_DATE": pd.to_datetime(tgt_date or date.today()),
        "DOW": (tgt_date or date.today()).weekday(),
        "MONTH": (tgt_date or date.today()).month,
    }
    out.update(_collect_side_features(hrow, hid, "HOME"))
    out.update(_collect_side_features(arow, aid, "AWAY"))

    # rest diffs
    out["REST_DIFF"] = out.get("REST_HOME", 0.0) - out.get("REST_AWAY", 0.0)
    out["B2B_DIFF"] = out.get("B2B_HOME", 0.0) - out.get("B2B_AWAY", 0.0)
    out["REST_R5_DIFF"] = out.get("REST_R5_HOME", 0.0) - out.get("REST_R5_AWAY", 0.0)
    out["B2B_RATE_R5_DIFF"] = out.get("B2B_RATE_R5_HOME", 0.0) - out.get("B2B_RATE_R5_AWAY", 0.0)

    # Elo expectation/diff
    Rh = float(out.get("HOME_ELO_PRE", 1500.0))
    Ra = float(out.get("AWAY_ELO_PRE", 1500.0))
    home_adv = 65.0
    Eh = 1.0 / (1.0 + 10 ** (-(Rh + home_adv - Ra) / 400.0))
    out["HOME_ELO_EXP"] = Eh
    out["ELO_DIFF"] = Rh - Ra

    # rolling DIFF columns, e.g., PTS_R10_HOME & _AWAY -> PTS_DIFF_R10
    keys = list(out.keys())
    for k in keys:
        if k.endswith("_HOME"):
            base = k[:-5]
            other = f"{base}_AWAY"
            if other in out and "_" in base:
                metric, suffix = base.rsplit("_", 1)
                out[f"{metric}_DIFF_{suffix}"] = float(out[k]) - float(out[other])

    return pd.DataFrame([out])

# ---------------- Model utils ----------------
def load_best_model():
    hgb = ART / "hgb_model.joblib"
    if hgb.exists():
        return "hgb", load(hgb), None
    lr, sc = ART / "model.joblib", ART / "scaler.joblib"
    if lr.exists() and sc.exists():
        return "lr", load(lr), load(sc)
    raise RuntimeError("No trained model found. Train first: python train_hgb.py (or train.py).")

def align_to_model(X: pd.DataFrame, model):
    """
    Ensure X has the exact same columns/order as during training.
    Uses sklearn's feature_names_in_ (available for models trained with pandas DataFrames).
    """
    if not hasattr(model, "feature_names_in_"):
        # best effort: keep as-is
        return X
    expected = list(model.feature_names_in_)
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    # drop any extra cols and order identically
    X = X[expected]
    # sklearn may expect numpy array shape
    return X

# ---------------- Lineups ----------------
def best_lineup_by_minutes(team_id: int, season_str: str, top_n: int = 5):
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str, per_mode_detailed="PerGame", team_id_nullable=team_id
        ).get_data_frames()[0]
        return df.sort_values("MIN", ascending=False)["PLAYER_NAME"].head(top_n).tolist()
    except Exception:
        return []

# ---------------- Predict core ----------------
def predict_from_text(query: str):
    home_txt, away_txt, dt, season_start, season_str = parse_query_text(query)
    hid, aid = fuzzy_team_id(home_txt), fuzzy_team_id(away_txt)

    # Load dataset and build engineered game-level frame
    team_rows = load_team_rows_from_csv(DATA_CSV)
    ds = build_dataset(team_rows)

    # Try exact row (past games)
    row = pd.DataFrame()
    venue_used = "requested"
    if dt:
        day_df = ds[ds["GAME_DATE"].dt.date == dt]
        row = day_df[(day_df["HOME_ID"] == hid) & (day_df["AWAY_ID"] == aid)]
    if row.empty and season_start:
        season_mask = ds["SEASON_ID"].astype(str).str.contains(str(season_start))
        season_df = ds[season_mask]
        cand = season_df[(season_df["HOME_ID"] == hid) & (season_df["AWAY_ID"] == aid)]
        if cand.empty:
            cand = season_df[(season_df["HOME_ID"] == aid) & (season_df["AWAY_ID"] == hid)]
            if not cand.empty:
                row = cand.sort_values("GAME_DATE").head(1).copy()
                venue_used = "reversed (proxy)"
        else:
            row = cand.sort_values("GAME_DATE").head(1).copy()

    # If still empty → synthesize (future/unscheduled)
    if row.empty:
        row = synthesize_matchup_row(ds, hid, aid, dt)
        venue_used = "synthesized (latest form)"

    # Build features and align to training feature order
    X = build_X_for_row(row)
    kind, model, scaler = load_best_model()
    X_aligned = align_to_model(X, model)

    # Predict
    if kind == "lr" and scaler is not None:
        Xs = scaler.transform(X_aligned)
        p_home = float(model.predict_proba(Xs)[:, 1][0])
    else:
        p_home = float(model.predict_proba(X_aligned)[:, 1][0])

    winner_id = hid if p_home >= 0.5 else aid
    winner = team_name(winner_id)

    # Season label for lineups
    if dt:
        season_for_lineups = f"{dt.year-1}-{str(dt.year)[-2:]}"
    elif season_str:
        season_for_lineups = season_str
    else:
        last_sid = str(team_rows["SEASON_ID"].astype(str).iloc[-1])
        try:
            start_guess = int(last_sid[-4:])
            season_for_lineups = f"{start_guess}-{str(start_guess+1)[-2:]}"
        except Exception:
            season_for_lineups = "2023-24"

    home5 = best_lineup_by_minutes(hid, season_for_lineups)
    away5 = best_lineup_by_minutes(aid, season_for_lineups)

    return {
        "parsed": {
            "home": home_txt,
            "away": away_txt,
            "date": dt.isoformat() if dt else None,
            "season": season_for_lineups,
            "venue_used": venue_used
        },
        "prediction": {
            "home_team": team_name(hid),
            "away_team": team_name(aid),
            "home_win_prob": p_home,
            "predicted_winner": winner
        },
        "suggested_lineups_by_minutes": {
            "home_top5": home5,
            "away_top5": away5
        }
    }

# ---------------- CLI ----------------
def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python predict.py "GSW vs LAL 2026-10-24"')
        print('  python predict.py "grizzlies @ warriors 2025-2026 season"')
        print('  python predict.py GSW LAL [2026-10-24]')
        sys.exit(1)

    query = parse_query(sys.argv[1:])
    try:
        out = predict_from_text(query)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
