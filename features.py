# features.py
import pandas as pd
import numpy as np

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
    - Rolling team form (R10, R30, SDT)
    - Rest and back to back
    - Game rows with home/away split
    - Elo, calendar, differentials
    """
    tr_roll = add_rolling(team_rows)
    tr_rest = add_rest(team_rows)

    game = team_to_game_rows(team_rows)

    base = ["TEAM_ID","GAME_ID"]
    roll_cols = [c for c in tr_roll.columns if c.endswith("_R10") or c.endswith("_R30") or c.endswith("_SDT")]

    home_roll = tr_roll[base + roll_cols].rename(columns={"TEAM_ID":"HOME_ID"})
    home_roll = home_roll.rename(columns={c: f"{c}_HOME" for c in roll_cols})
    away_roll = tr_roll[base + roll_cols].rename(columns={"TEAM_ID":"AWAY_ID"})
    away_roll = away_roll.rename(columns={c: f"{c}_AWAY" for c in roll_cols})

    df = game.merge(home_roll, on=["GAME_ID","HOME_ID"], how="left")
    df = df.merge(away_roll, on=["GAME_ID","AWAY_ID"], how="left")

    # rest features
    rest_cols = ["TEAM_ID","GAME_ID","REST_DAYS","B2B"]
    home_rest = tr_rest[rest_cols].rename(columns={"TEAM_ID":"HOME_ID","REST_DAYS":"REST_HOME","B2B":"B2B_HOME"})
    away_rest = tr_rest[rest_cols].rename(columns={"TEAM_ID":"AWAY_ID","REST_DAYS":"REST_AWAY","B2B":"B2B_AWAY"})
    df = df.merge(home_rest, on=["GAME_ID","HOME_ID"], how="left")
    df = df.merge(away_rest, on=["GAME_ID","AWAY_ID"], how="left")

    # Elo
    df = add_elo(df)

    # calendar
    df["DOW"] = df["GAME_DATE"].dt.dayofweek
    df["MONTH"] = df["GAME_DATE"].dt.month

    # drop early NaNs
    df = df.dropna().reset_index(drop=True)

    # differentials for each window (R10, R30, SDT)
    stat_list = ["PTS","AST","REB","TOV","FG_PCT","FG3_PCT","FT_PCT","PLUS_MINUS"]
    for m in stat_list:
        for suffix in ["R10","R30","SDT"]:
            df[f"{m}_DIFF_{suffix}"] = df[f"{m}_{suffix}_HOME"] - df[f"{m}_{suffix}_AWAY"]

    # Elo + rest diffs
    df["ELO_DIFF"] = df["HOME_ELO_PRE"] - df["AWAY_ELO_PRE"]
    df["REST_DIFF"] = df["REST_HOME"] - df["REST_AWAY"]
    df["B2B_DIFF"] = df["B2B_HOME"] - df["B2B_AWAY"]

    return df
