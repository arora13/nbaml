# predict.py
import re
import sys
import json
from pathlib import Path
from datetime import date
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import leaguedashplayerstats

from features import build_dataset  # your engineered features

ART = Path("artifacts")
DATA_CSV = Path("data/games.csv")

# -------- optional pretty printing --------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

# ---------------- Parsing ----------------
SEASON_RE = re.compile(r'(\d{4})\s*[-/]\s*(\d{4})')
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')

def parse_query(args: List[str]) -> str:
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

    clean_right = re.sub(SEASON_RE, '', re.sub(DATE_RE, '', right)).strip()
    home_name, away_name = (left, clean_right) if sep == 'vs' else (clean_right, left)
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
from joblib import load
ART = Path("artifacts")

def load_best_model():
    hgb = ART / "hgb_model.joblib"
    iso = ART / "isotonic.joblib"
    if hgb.exists():
        model = load(hgb)
        calibrator = load(iso) if iso.exists() else None
        return "hgb", model, None, calibrator
    lr, sc = ART / "model.joblib", ART / "scaler.joblib"
    if lr.exists() and sc.exists():
        return "lr", load(lr), load(sc), None
    raise RuntimeError("No trained model found. Train first: python train_hgb.py")

def align_to_model(X: pd.DataFrame, model):
    if not hasattr(model, "feature_names_in_"):
        return X
    expected = list(model.feature_names_in_)
    for col in expected:
        if col not in X.columns:
            X[col] = 0.0
    return X[expected]

# ---------------- Spread / margin estimate ----------------
def _learn_elo_to_margin(team_rows: pd.DataFrame, ds: pd.DataFrame) -> Tuple[float, float]:
    """Fit margin ~ a * ELO_DIFF + b, using home PLUS_MINUS from the home team row."""
    # home margin from team_rows (home row is the one with 'vs.')
    home_rows = team_rows[team_rows["MATCHUP"].astype(str).str.contains(r"vs\.", regex=True, na=False)].copy()
    margin_by_game = home_rows[["GAME_ID", "PLUS_MINUS"]].dropna()
    elo_by_game = ds[["GAME_ID", "ELO_DIFF"]].drop_duplicates()
    j = margin_by_game.merge(elo_by_game, on="GAME_ID", how="inner").dropna()
    if len(j) < 50:
        return (0.03, 0.0)  # fall back: ~0.03 pts of spread per Elo point (conservative)
    x = j["ELO_DIFF"].to_numpy()
    y = j["PLUS_MINUS"].to_numpy()
    a, b = np.polyfit(x, y, 1)  # slope, intercept
    return (float(a), float(b))

def predict_spread_from_row(row: pd.DataFrame, team_rows: pd.DataFrame, ds: pd.DataFrame) -> float:
    a, b = _learn_elo_to_margin(team_rows, ds)
    elo_diff = float(row.iloc[0].get("ELO_DIFF", 0.0))
    return a * elo_diff + b  # positive means HOME favored by that many points

# ---------------- Lineups & top scorer ----------------
def best_lineup_by_minutes(team_id: int, season_str: str, top_n: int = 5) -> List[str]:
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str, per_mode_detailed="PerGame", team_id_nullable=team_id
        ).get_data_frames()[0]
        return df.sort_values("MIN", ascending=False)["PLAYER_NAME"].head(top_n).tolist()
    except Exception:
        return []

def top_scorer_pick(team_id_a: int, team_id_b: int, season_str: str):
    """Return (player_name, team_name, ppg, reason_text)."""
    try:
        A = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str, per_mode_detailed="PerGame", team_id_nullable=team_id_a
        ).get_data_frames()[0][["PLAYER_NAME","PTS","MIN"]]
        B = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str, per_mode_detailed="PerGame", team_id_nullable=team_id_b
        ).get_data_frames()[0][["PLAYER_NAME","PTS","MIN"]]
        A["TEAM"] = team_name(team_id_a); B["TEAM"] = team_name(team_id_b)
        both = pd.concat([A,B], ignore_index=True).dropna(subset=["PTS"])
        best = both.sort_values(["PTS","MIN"], ascending=[False,False]).iloc[0]
        reason = f"{best['PLAYER_NAME']} leads these teams in season PPG ({best['PTS']:.1f})"
        return str(best["PLAYER_NAME"]), str(best["TEAM"]), float(best["PTS"]), reason
    except Exception:
        return None, None, None, "Season player stats unavailable (API)."

# ---------------- Reasons ----------------
def reasons_from_row(row: pd.Series) -> List[str]:
    rs = []
    elo_diff = float(row.get("ELO_DIFF", 0.0))
    if abs(elo_diff) >= 25:
        side = "home" if elo_diff > 0 else "away"
        rs.append(f"Elo edge favors {side} by {abs(elo_diff):.0f} rating pts.")

    rest_diff = float(row.get("REST_DIFF", 0.0))
    if abs(rest_diff) >= 0.5:
        side = "home" if rest_diff > 0 else "away"
        rs.append(f"{side.title()} rest advantage ~{abs(rest_diff):.1f} days.")

    b2b_diff = float(row.get("B2B_DIFF", 0.0))
    if abs(b2b_diff) >= 0.5:
        side = "home" if b2b_diff < 0 else "away"  # fewer B2Bs is better
        rs.append(f"Back-to-back differential favors {side}.")

    # check any rolling net/pts diffs present
    candidates = [c for c in row.index if ("NET" in c or "PTS" in c) and ("_DIFF_" in c)]
    if candidates:
        # pick largest absolute
        c = max(candidates, key=lambda k: abs(float(row.get(k, 0.0))))
        val = float(row.get(c, 0.0))
        side = "home" if val > 0 else "away"
        rs.append(f"Recent {c.replace('_',' ').lower()} leans {side} ({val:+.2f}).")
    return rs[:3]

# ---------------- Predict core ----------------
def predict_from_text(query: str):
    home_txt, away_txt, dt, season_start, season_str = parse_query_text(query)
    hid, aid = fuzzy_team_id(home_txt), fuzzy_team_id(away_txt)

    # Build engineered dataset from CSV
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

    # If still empty → synthesize from latest form (future/unscheduled)
    if row.empty:
        row = synthesize_matchup_row(ds, hid, aid, dt)
        venue_used = "synthesized (latest form)"

    # Build features and align to training feature order
    X = build_X_for_row(row.copy())
    kind, model, scaler, calibrator = load_best_model()   # <- note calibrator now
    X_aligned = align_to_model(X, model)

    # Predict prob (then apply isotonic if present)
    if kind == "lr" and scaler is not None:
        Xs = scaler.transform(X_aligned)
        p_home = float(model.predict_proba(Xs)[:, 1][0])
    else:
        p_home = float(model.predict_proba(X_aligned)[:, 1][0])
    if calibrator is not None:
        p_home = float(calibrator.predict([p_home])[0])

    # Predicted spread via Elo→margin regression
    spread = predict_spread_from_row(row, team_rows, ds)  # +: home favored

    # Top-scorer pick (season of the game or last season in data)
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

    ts_name, ts_team, ts_ppg, ts_reason = top_scorer_pick(hid, aid, season_for_lineups)

    # Winner
    winner_id = hid if p_home >= 0.5 else aid
    winner = team_name(winner_id)
    reasons = reasons_from_row(row.iloc[0])

    result = {
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
            "predicted_winner": winner,
            "predicted_spread_home": spread  # positive means home by N points
        },
        "top_scorer": {
            "player": ts_name, "team": ts_team, "season_ppg": ts_ppg, "why": ts_reason
        },
        "reasons": reasons
    }
    return result

# ---------------- CLI ----------------
def print_pretty(res: dict):
    ht = res["prediction"]["home_team"]
    at = res["prediction"]["away_team"]
    p = res["prediction"]["home_win_prob"]
    spread = res["prediction"]["predicted_spread_home"]
    winner = res["prediction"]["predicted_winner"]
    ts = res["top_scorer"]

    if not RICH:
        # plain text fallback
        print(f"{ht} vs {at}")
        print(f"Home win prob: {p*100:.1f}%")
        sign = "-" if spread >= 0 else "+"
        print(f"Predicted spread: {ht} {sign}{abs(spread):.1f}")
        print(f"Predicted winner: {winner}")
        if ts["player"]:
            print(f"Top scorer: {ts['player']} ({ts['team']}) ~ {ts['season_ppg']:.1f} ppg — {ts['why']}")
        if res["reasons"]:
            print("Reasons:")
            for r in res["reasons"]:
                print(f" - {r}")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Home", style="bold")
    table.add_column("Away", style="bold")
    table.add_column("Win Prob (Home)")
    table.add_column("Spread")
    table.add_row(
        ht, at, f"{p*100:.1f}%", f"{ht} {'-' if spread>=0 else '+'}{abs(spread):.1f}"
    )
    console.print(Panel(table, title="NBA ML — Game Prediction", expand=False))
    console.print(f"[bold]Predicted Winner:[/bold] {winner}")

    if ts["player"]:
        console.print(
            f"[bold]Top Scorer:[/bold] {ts['player']} ({ts['team']}) "
            f"[dim]~ {ts['season_ppg']:.1f} ppg[/dim]\n[dim]{ts['why']}[/dim]"
        )

    if res["reasons"]:
        bullets = "\n".join([f"• {r}" for r in res["reasons"]])
        console.print(Panel(bullets, title="Why the model leans this way", expand=False))

def main():
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python predict.py "GSW vs LAL 2026-10-24"')
        print('  python predict.py "grizzlies @ warriors 2025-2026 season"')
        print('  python predict.py GSW LAL [2026-10-24]')
        sys.exit(1)

    query = parse_query(sys.argv[1:])
    try:
        res = predict_from_text(query)
        if RICH:
            print_pretty(res)
        else:
            # default to pretty JSON if not using rich
            print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
