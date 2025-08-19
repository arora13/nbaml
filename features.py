# features.py
import numpy as np
import pandas as pd

_NUMERIC_EXCEPT = {
    "SEASON_ID", "GAME_ID", "MATCHUP", "WL",
    "TEAM_ABBREVIATION", "TEAM_NAME"
}
def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-text columns to numeric; parse GAME_DATE."""
    out = df.copy()
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    for c in out.columns:
        if c == "GAME_DATE" or c in _NUMERIC_EXCEPT:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def team_to_game_rows(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level rows (one row per team per game) into game-level rows
    (one row per game with home/away split and result).
    """
    tr = team_rows.copy()

    # safe home/away flag
    tr["MATCHUP"] = tr["MATCHUP"].astype(str)
    tr["IS_HOME"] = tr["MATCHUP"].str.contains(" vs. ", na=False)

    feats = ["PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]

    home = tr[tr["IS_HOME"]].copy()
    away = tr[~tr["IS_HOME"]].copy()

    game = home[["GAME_ID","TEAM_ID","GAME_DATE","SEASON_ID","WL"] + feats].merge(
        away[["GAME_ID","TEAM_ID"] + feats],
        on="GAME_ID",
        suffixes=("_HOME","_AWAY")
    )
    game = game.rename(columns={"TEAM_ID_HOME":"HOME_ID","TEAM_ID_AWAY":"AWAY_ID"})
    game["HOME_WIN"] = (game["WL"] == "W").astype(int)
    return game.sort_values("GAME_DATE").reset_index(drop=True)

def add_rolling(team_rows: pd.DataFrame, window=10) -> pd.DataFrame:
    """
    Rolling means per team, shifted (no leakage).
    Adds:
      - R10  (last 10 games)
      - R30  (last 30 games)
      - SDT  (season-to-date expanding mean)
    """
    tr = team_rows.sort_values(["TEAM_ID","GAME_DATE"]).copy()
    metrics = ["PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]

    # R10
    for m in metrics:
        tr[f"{m}_R10"] = (
            tr.groupby("TEAM_ID")[m]
              .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
        )
    # R30
    for m in metrics:
        tr[f"{m}_R30"] = (
            tr.groupby("TEAM_ID")[m]
              .transform(lambda s: s.shift(1).rolling(30, min_periods=5).mean())
        )
    # Season-to-date expanding mean (reset each season)
    tr["SEASON_KEY"] = tr["SEASON_ID"].astype(str)
    for m in metrics:
        tr[f"{m}_SDT"] = (
            tr.groupby(["TEAM_ID","SEASON_KEY"])[m]
              .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
        )

    tr["GAMES_PLAYED"] = tr.groupby("TEAM_ID").cumcount()
    return tr

def add_rest(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Rest features per team row:
      - REST_DAYS: days since previous game
      - B2B: 1 if REST_DAYS == 1, else 0
    """
    tr = team_rows.sort_values(["TEAM_ID","GAME_DATE"]).copy()
    last_date = tr.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    tr["REST_DAYS"] = (tr["GAME_DATE"] - last_date).dt.days
    tr["B2B"] = (tr["REST_DAYS"] == 1).astype("Int64").astype(float)
    return tr

def add_elo(game_rows: pd.DataFrame, k=20, home_adv=65) -> pd.DataFrame:
    """
    Simple Elo updated sequentially over time.
    Returns pre-game Elo and expected home win probability for each game.
    """
    df = game_rows.copy()
    teams = pd.unique(pd.concat([df["HOME_ID"], df["AWAY_ID"]]))
    elo = {int(t): 1500.0 for t in teams}
    pre = []

    for _, r in df.sort_values("GAME_DATE").iterrows():
        h, a = int(r["HOME_ID"]), int(r["AWAY_ID"])
        Rh, Ra = elo[h], elo[a]
        Eh = 1.0 / (1.0 + 10 ** (-(Rh + home_adv - Ra) / 400))
        pre.append({
            "GAME_ID": r["GAME_ID"],
            "HOME_ELO_PRE": Rh,
            "AWAY_ELO_PRE": Ra,
            "HOME_ELO_EXP": Eh
        })
        out = int(r["HOME_WIN"])
        elo[h] = Rh + k * (out - Eh)
        elo[a] = Ra + k * ((1 - out) - (1 - Eh))

    pre = pd.DataFrame(pre)
    return df.merge(pre, on="GAME_ID")

def build_dataset(team_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Master dataset builder:
      - Type coercion (prevents string arithmetic errors)
      - Rolling team form (R10, R30, SDT) via add_rolling(...)
      - Rest & back-to-back, plus rolling-5 rest/B2B rate
      - Game rows (home/away split) via team_to_game_rows(...)
      - Elo + calendar features
      - Home–away differentials
    """
    # 0) types first (avoids 'str - str' TypeError)
    tr = _coerce_numeric_columns(team_rows)

    # 1) per-team feature tables
    tr_roll = add_rolling(tr)          # expects numeric inputs; returns ..._R10/_R30/_SDT
    tr_rest = add_rest(tr)             # returns TEAM_ID, GAME_ID, REST_DAYS, B2B, (maybe GAME_DATE)

    # ensure GAME_DATE present in tr_rest for chronological rolling
    if "GAME_DATE" not in tr_rest.columns:
        tr_rest = tr_rest.merge(tr[["TEAM_ID", "GAME_ID", "GAME_DATE"]],
                                on=["TEAM_ID", "GAME_ID"], how="left")

    # rolling-5 rest features (per team, in time order)
    tr_rest = tr_rest.sort_values(["TEAM_ID", "GAME_DATE"])
    g = tr_rest.groupby("TEAM_ID", group_keys=False)
    tr_rest["REST_R5"]       = g["REST_DAYS"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    tr_rest["B2B_RATE_R5"]   = g["B2B"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    # 2) game-level scaffold (must include HOME_ID, AWAY_ID, GAME_ID, GAME_DATE, SEASON_ID, HOME_WIN)
    game = team_to_game_rows(tr).copy()

    # 3) join rolling form (home/away)
    base_keys = ["TEAM_ID", "GAME_ID"]
    roll_cols = [c for c in tr_roll.columns if c.endswith(("_R10", "_R30", "_SDT"))]

    home_roll = tr_roll[base_keys + roll_cols].rename(columns={"TEAM_ID": "HOME_ID"})
    home_roll = home_roll.rename(columns={c: f"{c}_HOME" for c in roll_cols})

    away_roll = tr_roll[base_keys + roll_cols].rename(columns={"TEAM_ID": "AWAY_ID"})
    away_roll = away_roll.rename(columns={c: f"{c}_AWAY" for c in roll_cols})

    df = game.merge(home_roll, on=["GAME_ID", "HOME_ID"], how="left")
    df = df.merge(away_roll, on=["GAME_ID", "AWAY_ID"], how="left")

    # 4) join rest/B2B (including rolling-5)
    rest_cols = ["TEAM_ID", "GAME_ID", "REST_DAYS", "B2B", "REST_R5", "B2B_RATE_R5"]
    home_rest = tr_rest[rest_cols].rename(columns={
        "TEAM_ID": "HOME_ID",
        "REST_DAYS": "REST_HOME",
        "B2B": "B2B_HOME",
        "REST_R5": "REST_R5_HOME",
        "B2B_RATE_R5": "B2B_RATE_R5_HOME",
    })
    away_rest = tr_rest[rest_cols].rename(columns={
        "TEAM_ID": "AWAY_ID",
        "REST_DAYS": "REST_AWAY",
        "B2B": "B2B_AWAY",
        "REST_R5": "REST_R5_AWAY",
        "B2B_RATE_R5": "B2B_RATE_R5_AWAY",
    })
    df = df.merge(home_rest, on=["GAME_ID", "HOME_ID"], how="left")
    df = df.merge(away_rest, on=["GAME_ID", "AWAY_ID"], how="left")

    # 5) Elo (adds HOME_ELO_PRE / AWAY_ELO_PRE; we also compute expectation & diff)
    df = add_elo(df)
    # if add_elo already provides HOME_ELO_EXP, this will overwrite with same logic
    Rh = pd.to_numeric(df.get("HOME_ELO_PRE", 1500), errors="coerce").fillna(1500)
    Ra = pd.to_numeric(df.get("AWAY_ELO_PRE", 1500), errors="coerce").fillna(1500)
    home_adv = 65.0  # ~standard NBA HFA in Elo points
    df["HOME_ELO_EXP"] = 1.0 / (1.0 + 10 ** (-(Rh + home_adv - Ra) / 400.0))
    df["ELO_DIFF"] = Rh - Ra

    # 6) calendar features
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["DOW"] = df["GAME_DATE"].dt.dayofweek
    df["MONTH"] = df["GAME_DATE"].dt.month

    # 7) engineered differentials (only if both sides exist)
    windows = ["R10", "R30", "SDT"]
    stats = ["PTS", "AST", "REB", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]
    for s in stats:
        for w in windows:
            hcol = f"{s}_{w}_HOME"
            acol = f"{s}_{w}_AWAY"
            dcol = f"{s}_DIFF_{w}"
            if hcol in df.columns and acol in df.columns:
                df[dcol] = pd.to_numeric(df[hcol], errors="coerce") - pd.to_numeric(df[acol], errors="coerce")

    # 8) rest/b2b differentials
    df["REST_DIFF"] = pd.to_numeric(df.get("REST_HOME"), errors="coerce") - pd.to_numeric(df.get("REST_AWAY"), errors="coerce")
    df["B2B_DIFF"] = pd.to_numeric(df.get("B2B_HOME"), errors="coerce") - pd.to_numeric(df.get("B2B_AWAY"), errors="coerce")
    df["REST_R5_DIFF"] = pd.to_numeric(df.get("REST_R5_HOME"), errors="coerce") - pd.to_numeric(df.get("REST_R5_AWAY"), errors="coerce")
    df["B2B_RATE_R5_DIFF"] = pd.to_numeric(df.get("B2B_RATE_R5_HOME"), errors="coerce") - pd.to_numeric(df.get("B2B_RATE_R5_AWAY"), errors="coerce")

    # 9) ensure target exists (team_to_game_rows should provide HOME_WIN)
    if "HOME_WIN" not in df.columns:
        # try to infer from per-game plus/minus if present
        if "HOME_PLUS_MINUS" in df.columns:
            df["HOME_WIN"] = (pd.to_numeric(df["HOME_PLUS_MINUS"], errors="coerce") > 0).astype(int)
        elif {"HOME_PTS", "AWAY_PTS"}.issubset(df.columns):
            df["HOME_WIN"] = (pd.to_numeric(df["HOME_PTS"], errors="coerce") >
                              pd.to_numeric(df["AWAY_PTS"], errors="coerce")).astype(int)
        else:
            # last resort: drop rows without target later; but keep column for pipeline
            df["HOME_WIN"] = np.nan

    # 10) final tidy: drop rows missing essentials
    essentials = ["GAME_ID", "HOME_ID", "AWAY_ID", "GAME_DATE", "HOME_ELO_PRE", "AWAY_ELO_PRE", "HOME_WIN"]
    df = df.dropna(subset=[c for c in essentials if c in df.columns]).reset_index(drop=True)

    return df
